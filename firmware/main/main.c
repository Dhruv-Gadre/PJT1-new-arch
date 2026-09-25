#include <math.h>
#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"

#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "nvs_flash.h"

#define CSI_SUBCARRIERS 12
#define CSI_QUEUE_LENGTH 32
#define WIFI_CONNECTED_BIT BIT0

static const char *TAG = "custom_csi";
static EventGroupHandle_t wifi_events;
static QueueHandle_t csi_queue;

struct csi_packet
{
    float amplitudes[CSI_SUBCARRIERS];
    float phases[CSI_SUBCARRIERS];
    uint32_t sequence;
};

static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START)
    {
        esp_wifi_connect();
    }
    else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED)
    {
        xEventGroupClearBits(wifi_events, WIFI_CONNECTED_BIT);
        ESP_LOGW(TAG, "Wi-Fi disconnected; retrying");
        esp_wifi_connect();
    }
    else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP)
    {
        const ip_event_got_ip_t *event = (const ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "Wi-Fi connected; IP address: " IPSTR, IP2STR(&event->ip_info.ip));
        xEventGroupSetBits(wifi_events, WIFI_CONNECTED_BIT);
    }
}

static void wifi_init_station(void)
{
    wifi_events = xEventGroupCreate();
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t init_config = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init_config));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                               &wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                               &wifi_event_handler, NULL));

    wifi_config_t wifi_config = {0};
    strncpy((char *)wifi_config.sta.ssid, CONFIG_CUSTOM_WIFI_SSID,
            sizeof(wifi_config.sta.ssid) - 1);
    strncpy((char *)wifi_config.sta.password, CONFIG_CUSTOM_WIFI_PASSWORD,
            sizeof(wifi_config.sta.password) - 1);
    wifi_config.sta.threshold.authmode = WIFI_AUTH_OPEN;
    wifi_config.sta.pmf_cfg.capable = true;
    wifi_config.sta.pmf_cfg.required = false;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_LOGI(TAG, "Connecting to configured Wi-Fi network...");
}

static void csi_callback(void *ctx, wifi_csi_info_t *info)
{
    if (info == NULL || info->buf == NULL || info->len < CSI_SUBCARRIERS * 2)
    {
        return;
    }

    struct csi_packet packet = {0};
    packet.sequence = info->first_word_invalid ? 0 : (uint32_t)info->rx_ctrl.timestamp;

    for (int subcarrier = 0; subcarrier < CSI_SUBCARRIERS; ++subcarrier)
    {
        int imag = (int8_t)info->buf[subcarrier * 2];
        int real = (int8_t)info->buf[subcarrier * 2 + 1];
        packet.amplitudes[subcarrier] = sqrtf((float)(real * real + imag * imag));
        packet.phases[subcarrier] = atan2f((float)imag, (float)real);
    }

    (void)xQueueSend(csi_queue, &packet, 0);
}

static void csi_print_task(void *arg)
{
    struct csi_packet packet;
    while (true)
    {
        if (xQueueReceive(csi_queue, &packet, portMAX_DELAY) != pdTRUE)
        {
            continue;
        }

        printf("{\"a\":[");
        for (int i = 0; i < CSI_SUBCARRIERS; ++i)
        {
            printf("%.4f%s", packet.amplitudes[i], i + 1 == CSI_SUBCARRIERS ? "" : ",");
        }
        printf("],\"p\":[");
        for (int i = 0; i < CSI_SUBCARRIERS; ++i)
        {
            printf("%.4f%s", packet.phases[i], i + 1 == CSI_SUBCARRIERS ? "" : ",");
        }
        printf("],\"ts\":%lu}\n", (unsigned long)packet.sequence);
        fflush(stdout);
    }
}

void app_main(void)
{
    esp_err_t nvs_result = nvs_flash_init();
    if (nvs_result == ESP_ERR_NVS_NO_FREE_PAGES ||
        nvs_result == ESP_ERR_NVS_NEW_VERSION_FOUND)
    {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    }

    csi_queue = xQueueCreate(CSI_QUEUE_LENGTH, sizeof(struct csi_packet));
    configASSERT(csi_queue != NULL);
    xTaskCreate(csi_print_task, "csi_print", 4096, NULL, 5, NULL);

    wifi_init_station();
    xEventGroupWaitBits(wifi_events, WIFI_CONNECTED_BIT, pdFALSE, pdTRUE,
                        portMAX_DELAY);

    wifi_csi_config_t csi_config = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = true,
        .ltf_merge_en = true,
        .channel_filter_en = false,
        .manu_scale = false,
        .shift = false,
    };
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(csi_callback, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
    ESP_LOGI(TAG, "CSI enabled; send local LAN traffic to this ESP32 IP to increase packet rate");
}

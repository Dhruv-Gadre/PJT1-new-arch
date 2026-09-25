/*
  Custom ESPectre CSI collector for Arduino IDE

  Hardware:
    - ESP32-S3
    - Tenda router providing a local 2.4 GHz Wi-Fi network

  The sketch connects to Wi-Fi, enables CSI, and prints one JSON packet per
  line over USB Serial. The Python collector in ../src reads these lines.

  Internet access is not required. Send local UDP traffic to the ESP32 IP
  from a laptop on the same router to increase the CSI packet rate.
*/

#include <Arduino.h>
#include <WiFi.h>
#include "esp_wifi.h"

// Edit these two values before uploading.
const char *WIFI_SSID = "ESPECTRE_LAB";
const char *WIFI_PASSWORD = "12345678";

// The callback uses the actual info->len value for each packet. This capacity
// keeps LLTF, HT-LTF, and STBC fields when the driver reports them together.
constexpr size_t MAX_SUBCARRIERS = 128;
constexpr size_t QUEUE_LENGTH = 16;

struct CsiPacket
{
    float amplitudes[MAX_SUBCARRIERS];
    float phases[MAX_SUBCARRIERS];
    uint16_t subcarrierCount;
    uint16_t rawLength;
    uint32_t sequence;
};

CsiPacket packetQueue[QUEUE_LENGTH];
volatile size_t queueHead = 0;
volatile size_t queueTail = 0;
volatile uint32_t packetSequence = 0;
portMUX_TYPE queueMux = portMUX_INITIALIZER_UNLOCKED;

void csiCallback(void *context, wifi_csi_info_t *info)
{
    if (info == nullptr || info->buf == nullptr || info->len < 2)
    {
        return;
    }

    portENTER_CRITICAL_ISR(&queueMux);
    size_t nextHead = (queueHead + 1) % QUEUE_LENGTH;
    if (nextHead == queueTail)
    {
        // Drop the oldest packet when the USB serial output cannot keep up.
        queueTail = (queueTail + 1) % QUEUE_LENGTH;
    }

    CsiPacket &packet = packetQueue[queueHead];
    size_t availableSubcarriers = static_cast<size_t>(info->len) / 2;
    packet.subcarrierCount = min(availableSubcarriers, MAX_SUBCARRIERS);
    packet.rawLength = static_cast<uint16_t>(info->len);
    packet.sequence = ++packetSequence;
    for (size_t subcarrier = 0; subcarrier < packet.subcarrierCount; ++subcarrier)
    {
        int imag = static_cast<int8_t>(info->buf[subcarrier * 2]);
        int real = static_cast<int8_t>(info->buf[subcarrier * 2 + 1]);
        packet.amplitudes[subcarrier] = sqrtf(static_cast<float>(real * real + imag * imag));
        packet.phases[subcarrier] = atan2f(static_cast<float>(imag), static_cast<float>(real));
    }

    queueHead = nextHead;
    portEXIT_CRITICAL_ISR(&queueMux);
}

bool popPacket(CsiPacket &packet)
{
    bool available = false;
    portENTER_CRITICAL(&queueMux);
    if (queueTail != queueHead)
    {
        packet = packetQueue[queueTail];
        queueTail = (queueTail + 1) % QUEUE_LENGTH;
        available = true;
    }
    portEXIT_CRITICAL(&queueMux);
    return available;
}

void connectToWiFi()
{
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    Serial.printf("Connecting to Wi-Fi SSID: %s\n", WIFI_SSID);
    while (WiFi.status() != WL_CONNECTED)
    {
        delay(500);
        Serial.print(".");
    }

    Serial.println();
    Serial.print("Wi-Fi connected; IP address: ");
    Serial.println(WiFi.localIP());
    Serial.println("Send local UDP traffic to this IP to increase CSI packet rate.");
}

void enableCsi()
{
    wifi_csi_config_t csiConfig = {};
    csiConfig.lltf_en = true;
    csiConfig.htltf_en = true;
    csiConfig.stbc_htltf2_en = true;
    csiConfig.ltf_merge_en = true;
    csiConfig.channel_filter_en = false;
    csiConfig.manu_scale = false;
    csiConfig.shift = false;

    esp_err_t result = esp_wifi_set_csi_config(&csiConfig);
    if (result != ESP_OK)
    {
        Serial.printf("ERROR: esp_wifi_set_csi_config failed: %s\n", esp_err_to_name(result));
        return;
    }

    result = esp_wifi_set_csi_rx_cb(csiCallback, nullptr);
    if (result != ESP_OK)
    {
        Serial.printf("ERROR: esp_wifi_set_csi_rx_cb failed: %s\n", esp_err_to_name(result));
        return;
    }

    result = esp_wifi_set_csi(true);
    if (result != ESP_OK)
    {
        Serial.printf("ERROR: esp_wifi_set_csi failed: %s\n", esp_err_to_name(result));
        return;
    }

    Serial.println("CSI enabled.");
}

void printPacket(const CsiPacket &packet)
{
    Serial.print("{\"a\":[");
    for (size_t i = 0; i < packet.subcarrierCount; ++i)
    {
        Serial.printf("%.4f%s", packet.amplitudes[i], i + 1 == packet.subcarrierCount ? "" : ",");
    }

    Serial.print("],\"p\":[");
    for (size_t i = 0; i < packet.subcarrierCount; ++i)
    {
        Serial.printf("%.4f%s", packet.phases[i], i + 1 == packet.subcarrierCount ? "" : ",");
    }
    Serial.printf("],\"n\":%u,\"ts\":%lu}\n",
                  packet.subcarrierCount, static_cast<unsigned long>(packet.sequence));
}

void setup()
{
    Serial.begin(921600);
    delay(1000);
    Serial.println();
    Serial.println("Custom ESPectre CSI collector starting");

    connectToWiFi();
    enableCsi();
}

void loop()
{
    CsiPacket packet;
    static bool shapeReported = false;
    while (popPacket(packet))
    {
        if (!shapeReported)
        {
            Serial.printf("CSI buffer: %u bytes (%u IQ pairs); retaining %u subcarriers\n",
                          packet.rawLength, packet.rawLength / 2, packet.subcarrierCount);
            shapeReported = true;
        }
        printPacket(packet);
    }
    delay(1);
}

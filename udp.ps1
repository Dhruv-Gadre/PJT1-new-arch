$espIp = "192.168.0.100"
$port = 5005
$intervalMs = 10

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " ESPectre CSI Traffic Generator" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Target ESP32 IP : $espIp"
Write-Host "Target UDP port : $port"
Write-Host "Packet interval : $intervalMs ms"
Write-Host ""

Write-Host "Creating UDP client..." -ForegroundColor Yellow

$udp = [System.Net.Sockets.UdpClient]::new()

$bytes = [System.Text.Encoding]::ASCII.GetBytes("csi")

Write-Host "UDP client created." -ForegroundColor Green
Write-Host ""
Write-Host "Starting packet transmission..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop."
Write-Host ""

$count = 0

try {
    while ($true) {

        [void]$udp.Send(
            $bytes,
            $bytes.Length,
            $espIp,
            $port
        )

        $count++

        if ($count % 100 -eq 0) {
            $time = Get-Date -Format "HH:mm:ss"

            Write-Host "[$time] Sent $count UDP packets to $espIp`:$port" `
                -ForegroundColor Green
        }

        Start-Sleep -Milliseconds $intervalMs
    }
}
catch {
    Write-Host ""
    Write-Host "ERROR:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}
finally {
    $udp.Dispose()

    Write-Host ""
    Write-Host "UDP client stopped." -ForegroundColor Yellow
    Write-Host "Total packets sent: $count"
}
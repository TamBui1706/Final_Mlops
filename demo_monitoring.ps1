<#
.SYNOPSIS
    Demo Prometheus & Grafana với auto-generated traffic

.DESCRIPTION
    Script này sẽ mở 2 terminal windows:
    - Window 1: Tự động generate traffic liên tục
    - Window 2: Theo dõi metrics realtime
    
    Bạn có thể mở Grafana (http://localhost:3000) hoặc 
    Prometheus (http://localhost:9090) để xem dashboard

.EXAMPLE
    .\demo_monitoring.ps1
    
.EXAMPLE
    .\demo_monitoring.ps1 -Interval 2 -TrafficInterval 1.5
#>

param(
    [Parameter(HelpMessage="Refresh interval cho metrics monitor (seconds)")]
    [int]$Interval = 5,
    
    [Parameter(HelpMessage="Interval giữa các requests (seconds)")]
    [double]$TrafficInterval = 2.0,
    
    [Parameter(HelpMessage="Tự động mở browser Grafana & Prometheus")]
    [switch]$OpenBrowser
)

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "🚀 STARTING MONITORING DEMO" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

# Kiểm tra Docker containers
Write-Host "📦 Checking Docker containers..." -ForegroundColor Yellow
$containers = docker-compose ps --services --filter "status=running" 2>$null

if ($LASTEXITCODE -ne 0 -or -not $containers) {
    Write-Host "❌ Docker containers are not running!" -ForegroundColor Red
    Write-Host "💡 Run: docker-compose up -d" -ForegroundColor Yellow
    exit 1
}

$required = @("api", "prometheus", "grafana")
$running = $containers -split "`n" | Where-Object { $_ -in $required }

if ($running.Count -ne $required.Count) {
    Write-Host "❌ Missing required containers!" -ForegroundColor Red
    Write-Host "Required: $($required -join ', ')" -ForegroundColor Yellow
    Write-Host "Running: $($running -join ', ')" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ All required containers are running" -ForegroundColor Green
Write-Host ""

# Mở browsers nếu có flag
if ($OpenBrowser) {
    Write-Host "🌐 Opening browsers..." -ForegroundColor Yellow
    Start-Process "http://localhost:3000"  # Grafana
    Start-Process "http://localhost:9090"  # Prometheus
    Start-Sleep -Seconds 2
}

Write-Host "📊 URLs:" -ForegroundColor Cyan
Write-Host "   • Grafana:    http://localhost:3000 (admin/admin)" -ForegroundColor White
Write-Host "   • Prometheus: http://localhost:9090" -ForegroundColor White
Write-Host "   • API Docs:   http://localhost:8000/docs" -ForegroundColor White
Write-Host ""

# Terminal 1: Generate traffic
Write-Host "🚀 Starting traffic generator in new window..." -ForegroundColor Yellow
$trafficCmd = "python auto_generate_traffic.py --interval $TrafficInterval"
Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
`$Host.UI.RawUI.WindowTitle = 'Traffic Generator'
Write-Host '🚀 AUTO TRAFFIC GENERATOR' -ForegroundColor Green
Write-Host '=' * 80 -ForegroundColor Cyan
Write-Host 'Generating requests every $TrafficInterval seconds' -ForegroundColor Yellow
Write-Host 'Press Ctrl+C to stop' -ForegroundColor Yellow
Write-Host '=' * 80 -ForegroundColor Cyan
Write-Host ''
$trafficCmd
"@

Start-Sleep -Seconds 1

# Terminal 2: Watch metrics
Write-Host "📈 Starting metrics monitor in new window..." -ForegroundColor Yellow
$metricsCmd = "python watch_metrics.py --interval $Interval"
Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
`$Host.UI.RawUI.WindowTitle = 'Metrics Monitor'
Write-Host '📊 REALTIME METRICS MONITOR' -ForegroundColor Green
Write-Host '=' * 80 -ForegroundColor Cyan
Write-Host 'Refreshing every $Interval seconds' -ForegroundColor Yellow
Write-Host 'Press Ctrl+C to stop' -ForegroundColor Yellow
Write-Host '=' * 80 -ForegroundColor Cyan
Write-Host ''
$metricsCmd
"@

Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "✅ DEMO STARTED!" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""
Write-Host "📋 Next Steps:" -ForegroundColor Cyan
Write-Host "   1. Watch metrics update in the Metrics Monitor window" -ForegroundColor White
Write-Host "   2. Open Grafana (http://localhost:3000)" -ForegroundColor White
Write-Host "   3. Create dashboard with queries from DEMO.md section 8.2" -ForegroundColor White
Write-Host "   4. Watch dashboard update in realtime!" -ForegroundColor White
Write-Host ""
Write-Host "💡 To stop: Close both terminal windows or press Ctrl+C in each" -ForegroundColor Yellow
Write-Host ""

# VIDDHANA Project Startup Script
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  VIDDHANA Development Environment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Set location to project root
Set-Location "K:\Viddhana_git\docs\viddhana"

# Check if dependencies are installed
if (-Not (Test-Path "node_modules")) {
    Write-Host "[1/4] Installing dependencies..." -ForegroundColor Yellow
    pnpm install
} else {
    Write-Host "[1/4] Dependencies already installed ✓" -ForegroundColor Green
}

Write-Host ""
Write-Host "[2/4] Checking Docker services..." -ForegroundColor Yellow
$dockerRunning = docker info 2>$null
if ($?) {
    Write-Host "Docker is running. Starting infrastructure services..." -ForegroundColor Green
    docker compose up -d
    Write-Host "Infrastructure services started ✓" -ForegroundColor Green
} else {
    Write-Host "Docker is not running. Skipping infrastructure services..." -ForegroundColor Yellow
    Write-Host "Note: Some features may not work without Docker services" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[3/4] Starting development servers..." -ForegroundColor Yellow
Write-Host ""

# Start development servers in new terminal windows
Write-Host "Starting Documentation Server (Port 3030)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd K:\Viddhana_git\docs\viddhana\docs; npm start"

Start-Sleep -Seconds 2

Write-Host "Starting API Server (Port 3000)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd K:\Viddhana_git\docs\viddhana\packages\api-server; pnpm dev"

Start-Sleep -Seconds 2

Write-Host "Starting JavaScript SDK in watch mode..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd K:\Viddhana_git\docs\viddhana\packages\sdks\javascript; pnpm dev"

Write-Host ""
Write-Host "[4/4] All services started!" -ForegroundColor Green
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Services Running:" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "📚 Documentation:  http://localhost:3030" -ForegroundColor White
Write-Host "🚀 API Server:     http://localhost:3000" -ForegroundColor White
Write-Host "🔧 SDK:            Building in watch mode" -ForegroundColor White
Write-Host ""
Write-Host "Docker Services (if running):" -ForegroundColor White
Write-Host "  - PostgreSQL:    localhost:5432" -ForegroundColor Gray
Write-Host "  - Redis:         localhost:6379" -ForegroundColor Gray
Write-Host "  - Hardhat:       localhost:8545" -ForegroundColor Gray
Write-Host "  - Kafka:         localhost:9092" -ForegroundColor Gray
Write-Host "  - MQTT:          localhost:1883" -ForegroundColor Gray
Write-Host "  - MLflow:        http://localhost:5000" -ForegroundColor Gray
Write-Host ""
Write-Host "Press any key to continue..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")


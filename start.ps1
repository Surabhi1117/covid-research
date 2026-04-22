Write-Host "🚀 Starting COVID-19 Research Platform..." -ForegroundColor Cyan

# Cleanup old processes
Write-Host "🧹 Cleaning up old processes..." -ForegroundColor Gray
Get-NetTCPConnection -LocalPort 8000, 8501, 8506 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }

# Check if MongoDB is running
if (!(Get-NetTCPConnection -LocalPort 27017 -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Error: MongoDB is not running on localhost:27017" -ForegroundColor Red
    exit 1
}

# Start Backend in a new window
Write-Host "🌐 Starting Backend (FastAPI) on http://localhost:8000..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m api.main"

# Start Frontend in a new window
Write-Host "🎨 Starting Frontend (Streamlit) on http://localhost:8501..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "streamlit run frontend/app.py"

Write-Host "✅ Both services are starting! Please wait a few seconds for the models to load." -ForegroundColor Yellow

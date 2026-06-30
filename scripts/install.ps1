# Install Stereo_Vision dependencies (Windows PowerShell).

Write-Host "=== Stereo_Vision install ==="

# Create venv
if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

# Activate & install deps
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host ""
Write-Host "=== ZED SDK ===" -ForegroundColor Yellow
Write-Host "pyzed.sl is NOT in PyPI. After installing ZED SDK from stereolabs.com,"
Write-Host "add the ZED Python API to PYTHONPATH:"
Write-Host '  $env:PYTHONPATH = "$env:PYTHONPATH;C:\Program Files\Stereolabs\ZED\scripts"'
Write-Host ""
Write-Host "=== TLS Certificate ===" -ForegroundColor Yellow
Write-Host "Generate local cert:"
Write-Host "  .\config\gen_self_signed_cert.ps1"
Write-Host ""
Write-Host "=== Run ===" -ForegroundColor Yellow
Write-Host "  .\.venv\Scripts\python.exe main.py"

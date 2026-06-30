# Generate self-signed TLS certificate for local HTTPS (Windows).

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$CONFIG_DIR = Split-Path -Parent $SCRIPT_DIR

if (-not (Get-Command openssl -ErrorAction SilentlyContinue)) {
    Write-Error "openssl not found. Install OpenSSL or use Git Bash."
    exit 1
}

$cert = Join-Path $CONFIG_DIR "cert.pem"
$key = Join-Path $CONFIG_DIR "key.pem"

if ((Test-Path $cert) -and (Test-Path $key)) {
    Write-Host "Certificate already exists at $cert"
    exit 0
}

Write-Host "Generating self-signed certificate in $CONFIG_DIR/..."

# Use Git Bash's openssl path if needed
$OPENSSL = if (Get-Command openssl.exe -ErrorAction SilentlyContinue) { "openssl.exe" } else { "openssl" }

& $OPENSSL req -x509 -nodes -days 365 `
    -newkey rsa:2048 `
    -keyout $key `
    -out $cert `
    -subj "/CN=localhost/O=StereoVision/C=FI"

Write-Host "Done: $cert and $key"

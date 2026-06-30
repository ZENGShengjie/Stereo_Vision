# Install Stereo_Vision dependencies.

echo "=== Stereo_Vision install ==="

# Create venv if not exists
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Activate
source .venv/bin/activate

# Install Python deps
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=== ZED SDK ==="
echo "pyzed.sl is NOT in PyPI. After installing ZED SDK from"
echo "https://www.stereolabs.com/developers, set PYTHONPATH:"
echo "  Linux/macOS:  export PYTHONPATH=\$PYTHONPATH:/usr/local/zed"
echo "  Windows:      set PYTHONPATH=%PYTHONPATH%;C:\Program Files\Stereolabs\ZED\scripts"
echo ""
echo "=== TLS Certificate ==="
echo "Generate local cert:"
echo "  bash ./config/gen_self_signed_cert.sh"
echo "  # or on Windows (with OpenSSL):"
echo "  powershell -File ./config/gen_self_signed_cert.ps1"
echo ""
echo "=== Run ==="
echo "  .venv/bin/python main.py"
echo "  # or:"
echo "  python main.py"

#!/usr/bin/env bash
# Generate self-signed TLS certificate for local HTTPS.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$(dirname "$SCRIPT_DIR")/config"

mkdir -p "$CONFIG_DIR"

CERT="$CONFIG_DIR/cert.pem"
KEY="$CONFIG_DIR/key.pem"

if [ -f "$CERT" ] && [ -f "$KEY" ]; then
    echo "Certificate already exists at $CERT"
    exit 0
fi

echo "Generating self-signed certificate in $CONFIG_DIR/..."

openssl req -x509 -nodes -days 365 \
    -newkey rsa:2048 \
    -keyout "$KEY" \
    -out "$CERT" \
    -subj "/CN=localhost/O=StereoVision/C=FI" \
    2>/dev/null

echo "Done: $CERT and $KEY"

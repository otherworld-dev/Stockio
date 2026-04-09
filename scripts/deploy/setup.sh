#!/usr/bin/env bash
# Stockio LXC Container Setup Script
# Run as root on a fresh Debian/Ubuntu LXC container
set -euo pipefail

APP_DIR="/opt/stockio"
LOG_DIR="/var/log/stockio"
SERVICE_USER="stockio"

echo "=== Stockio LXC Deployment ==="

# 1. System packages
echo "[1/8] Installing system packages..."
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip git

# 2. Create service user
echo "[2/8] Creating service user..."
if ! id "$SERVICE_USER" &>/dev/null; then
    useradd --system --shell /usr/sbin/nologin --home-dir "$APP_DIR" "$SERVICE_USER"
fi
# Add calling user to stockio group for CLI access
CALLING_USER="${SUDO_USER:-}"
if [ -n "$CALLING_USER" ] && [ "$CALLING_USER" != "root" ]; then
    usermod -aG "$SERVICE_USER" "$CALLING_USER"
fi

# 3. Clone or update repo
echo "[3/8] Setting up application..."
mkdir -p "$APP_DIR" "$LOG_DIR"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Copy project files
cp -r "$REPO_DIR/src" "$APP_DIR/"
cp -r "$REPO_DIR/config" "$APP_DIR/"
cp "$REPO_DIR/pyproject.toml" "$APP_DIR/"
mkdir -p "$APP_DIR/data" "$APP_DIR/data/models" "$APP_DIR/models"

# 4. Create .env from example if missing
if [ ! -f "$APP_DIR/.env" ]; then
    cp "$REPO_DIR/.env.example" "$APP_DIR/.env"
    echo "    Created $APP_DIR/.env — edit with your settings"
fi

# 5. Python venv + dependencies (NO torch/transformers)
echo "[4/8] Setting up Python environment..."
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install --upgrade pip setuptools wheel -q
echo "    Installing dependencies..."
"$APP_DIR/.venv/bin/pip" install -e "$APP_DIR" -q

# 6. Permissions
echo "[5/8] Setting permissions..."
chown -R "$SERVICE_USER:$SERVICE_USER" "$APP_DIR" "$LOG_DIR"
chmod -R g+w "$APP_DIR/data" "$LOG_DIR"
chmod g+s "$APP_DIR/data"

# 7. Sudoers for web dashboard bot control
echo "[6/8] Configuring service permissions..."
cat > /etc/sudoers.d/stockio <<SUDOERS
stockio ALL=(root) NOPASSWD: /usr/bin/systemctl start stockio, /usr/bin/systemctl stop stockio, /usr/bin/systemctl restart stockio
SUDOERS
chmod 440 /etc/sudoers.d/stockio

# 8. Install systemd services
echo "[7/8] Installing systemd services..."
cp "$REPO_DIR/scripts/deploy/stockio.service" /etc/systemd/system/
cp "$REPO_DIR/scripts/deploy/stockio-web.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable stockio.service
systemctl enable stockio-web.service

# 9. Add to PATH + start web
echo "[8/8] Starting services..."
echo "export PATH=\"$APP_DIR/.venv/bin:\$PATH\"" > /etc/profile.d/stockio.sh
chmod +x /etc/profile.d/stockio.sh
systemctl start stockio-web.service

IP=$(hostname -I | awk '{print $1}')
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Dashboard:  http://${IP}:5000"
echo "Config:     $APP_DIR/.env"
echo "Database:   $APP_DIR/data/stockio.db"
echo "Logs:       journalctl -u stockio -f"
echo ""
echo "Commands:"
echo "  systemctl start stockio         # Start trading bot"
echo "  systemctl stop stockio          # Stop trading bot"
echo "  systemctl status stockio-web    # Check dashboard"
echo "  journalctl -u stockio -f        # Follow bot logs"
echo ""
echo "The bot runs in paper mode by default (Yahoo Finance, no signup needed)."
echo "Add OANDA credentials to $APP_DIR/.env for live trading."

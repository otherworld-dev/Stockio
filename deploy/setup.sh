#!/usr/bin/env bash
# Stockio LXC Container Setup Script
# Run as root on a fresh Debian/Ubuntu LXC container
set -euo pipefail

APP_DIR="/opt/stockio"
LOG_DIR="/var/log/stockio"
SERVICE_USER="stockio"

echo "=== Stockio LXC Deployment ==="

# 1. System packages
echo "[1/7] Installing system packages..."
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip git

# 2. Create service user and group
echo "[2/7] Creating service user..."
if ! id "$SERVICE_USER" &>/dev/null; then
    useradd --system --shell /usr/sbin/nologin --home-dir "$APP_DIR" "$SERVICE_USER"
fi
# Add the calling user to the stockio group so they can run CLI commands
# (e.g. stockio train) without permission errors
CALLING_USER="${SUDO_USER:-}"
if [ -n "$CALLING_USER" ] && [ "$CALLING_USER" != "root" ]; then
    usermod -aG "$SERVICE_USER" "$CALLING_USER"
fi

# 3. Set up application directory
echo "[3/7] Setting up application..."
mkdir -p "$APP_DIR" "$LOG_DIR"

# Copy project files (assumes this script is run from the repo root)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

cp -r "$REPO_DIR/src" "$APP_DIR/"
cp -r "$REPO_DIR/data" "$APP_DIR/"
cp "$REPO_DIR/requirements.txt" "$APP_DIR/"
cp "$REPO_DIR/setup.py" "$APP_DIR/"

# 4. Create .env from example if it doesn't exist
if [ ! -f "$APP_DIR/.env" ]; then
    cp "$REPO_DIR/.env.example" "$APP_DIR/.env"
    echo "    Created $APP_DIR/.env — edit this file with your settings"
fi

# 5. Python virtual environment and dependencies
echo "[4/7] Setting up Python environment..."
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install --upgrade pip setuptools wheel -q
echo "    Installing PyTorch (CPU-only, ~200MB)..."
"$APP_DIR/.venv/bin/pip" install torch --index-url https://download.pytorch.org/whl/cpu
echo "    Installing remaining dependencies..."
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt"
echo "    Installing stockio package..."
"$APP_DIR/.venv/bin/pip" install -e "$APP_DIR" -q

# 6. Set permissions (group-writable data dir so CLI users can train/write)
echo "[5/7] Setting permissions..."
chown -R "$SERVICE_USER:$SERVICE_USER" "$APP_DIR" "$LOG_DIR"
chmod -R g+w "$APP_DIR/data" "$LOG_DIR"
chmod g+s "$APP_DIR/data" "$APP_DIR/data/models"

# 7. Install systemd services
echo "[6/7] Installing systemd services..."
cp "$REPO_DIR/deploy/stockio-web.service" /etc/systemd/system/
cp "$REPO_DIR/deploy/stockio-bot.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable stockio-web.service
systemctl enable stockio-bot.service

# 8. Start services
echo "[7/7] Starting services..."
systemctl start stockio-web.service
echo "    Web dashboard started on http://$(hostname -I | awk '{print $1}'):5000"

# 9. Add stockio venv to system PATH
echo "[8/8] Adding stockio to PATH..."
echo "export PATH=\"$APP_DIR/.venv/bin:\$PATH\"" > /etc/profile.d/stockio.sh
chmod +x /etc/profile.d/stockio.sh

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Commands:"
echo "  systemctl start stockio-bot     # Start the trading bot"
echo "  systemctl stop stockio-bot      # Stop the trading bot"
echo "  systemctl status stockio-web    # Check web dashboard status"
echo "  journalctl -u stockio-bot -f    # Follow bot logs"
echo "  journalctl -u stockio-web -f    # Follow web logs"
echo ""
echo "Configuration: $APP_DIR/.env"
echo "Database:      $APP_DIR/data/stockio.db"
echo "Logs:          $LOG_DIR/"
echo ""
echo "IMPORTANT: Edit $APP_DIR/.env before starting the bot."
echo "  The bot starts in paper trading mode by default (safe)."
echo "  To start the bot: systemctl start stockio-bot"

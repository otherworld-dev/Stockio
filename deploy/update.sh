#!/usr/bin/env bash
# Stockio Update Script
# Run as root from the repo root to deploy new code to /opt/stockio
set -euo pipefail

APP_DIR="/opt/stockio"
SERVICE_USER="stockio"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Stockio Update ==="

# 1. Copy updated source and config files
echo "[1/4] Copying updated files..."
cp -r "$REPO_DIR/src" "$APP_DIR/"
cp "$REPO_DIR/requirements.txt" "$APP_DIR/"
cp "$REPO_DIR/setup.py" "$APP_DIR/"

# Copy deploy files (services may have changed)
cp "$REPO_DIR/deploy/stockio-web.service" /etc/systemd/system/
cp "$REPO_DIR/deploy/stockio-bot.service" /etc/systemd/system/

# 2. Ensure sudoers rule for web → bot control
if [ ! -f /etc/sudoers.d/stockio ]; then
    echo "    Adding sudoers rule for bot management..."
    cat > /etc/sudoers.d/stockio <<SUDOERS
stockio ALL=(root) NOPASSWD: /usr/bin/systemctl start stockio-bot, /usr/bin/systemctl stop stockio-bot, /usr/bin/systemctl restart stockio-bot
SUDOERS
    chmod 440 /etc/sudoers.d/stockio
fi

# 3. Fix permissions and reinstall package
echo "[3/4] Updating permissions and package..."
chown -R "$SERVICE_USER:$SERVICE_USER" "$APP_DIR/src"
"$APP_DIR/.venv/bin/pip" install -e "$APP_DIR" -q
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt" -q

# 4. Restart services
echo "[4/4] Restarting services..."
systemctl daemon-reload
systemctl restart stockio-web.service
if systemctl is-active --quiet stockio-bot.service; then
    systemctl restart stockio-bot.service
    echo "    Bot restarted"
fi

echo ""
echo "=== Update Complete ==="
echo "Web dashboard: http://$(hostname -I | awk '{print $1}'):5000"

#!/usr/bin/env bash
# Update Stockio on the server after git pull
set -euo pipefail

APP_DIR="/opt/stockio"
REPO_DIR="$(cd "$(dirname "$(dirname "$(dirname "$0")")")" && pwd)"

echo "=== Updating Stockio ==="

# Copy updated files
cp -r "$REPO_DIR/src" "$APP_DIR/"
cp -r "$REPO_DIR/config" "$APP_DIR/"
cp "$REPO_DIR/pyproject.toml" "$APP_DIR/"

# Update dependencies
"$APP_DIR/.venv/bin/pip" install -e "$APP_DIR" -q

# Fix permissions
chown -R stockio:stockio "$APP_DIR"

# Restart services
systemctl restart stockio-web
systemctl restart stockio 2>/dev/null || true

echo "Done. Dashboard restarted."

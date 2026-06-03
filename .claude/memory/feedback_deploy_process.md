---
name: Deploy process
description: How to deploy Stockio changes to the production server - never manually copy files
type: feedback
---

Always deploy via the update script, never manually copy files to /opt/stockio.

Deploy command:
```
cd ~/stockio && git pull && sudo bash scripts/deploy/update.sh
```

This copies src/, config/, pyproject.toml to /opt/stockio, installs dependencies, fixes permissions, and restarts both stockio and stockio-web services.

**Why:** We fell into manually copying files which caused features to not be deployed. The update script handles everything correctly including permissions (chown stockio:stockio) and service restarts.

**How to apply:** After committing and pushing changes, tell the user to run the deploy command above. Never suggest `sudo cp` for individual files.

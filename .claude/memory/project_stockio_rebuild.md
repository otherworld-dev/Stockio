---
name: Stockio project state
description: Merged trading bot with Flask dashboard, OANDA broker, LightGBM ML, Claude Haiku sentiment
type: project
---

Stockio was rebuilt by merging two branches: an old mature branch (Flask dashboard, multi-broker, ~8000 LOC) and a clean master rebuild (~2100 LOC). The old branch crashed from torch/transformers OOM.

**Current state (3100 LOC, 15 commits):**
- OANDA broker with tenacity retries
- LightGBM ever-learning model with outcome tracking
- Claude Haiku API for sentiment (replaces FinBERT, ~0MB RAM)
- Flask dashboard (dark mode, Chart.js, real-time)
- SQLite persistence (trades, snapshots, bot logs)
- CLI: `stockio run`, `stockio web`, `stockio status`, `stockio history`, `stockio pnl`, `stockio signals`
- Risk management with circuit breakers, drawdown kill switch
- Paper + Live instances simultaneously
- systemd deploy scripts

**Old branch ref:** `origin/claude/ai-stock-trading-bot-EG41T` on `otherworld-dev/Stockio` GitHub repo

**How to apply:** torch/transformers MUST NEVER be added back (caused OOM). Use Claude Haiku for NLP. User waiting for OANDA UK account approval.

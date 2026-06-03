---
name: Strategy competition next steps
description: After collecting enough trade data from the 4 competing strategies, decide how to use results to improve trading
type: project
---

4 strategies (trend, meanrev, momentum, llm) are running on separate OANDA practice sub-accounts since 2026-04-30, each starting with £100,000.

**Why:** Determine which scoring approach performs best and use the data to improve overall trading.

**How to apply:** After a few hundred trades per strategy (est. 2-3 weeks), evaluate:
1. Kill underperforming strategies, reallocate capital
2. Blend signals from winning strategies into a combined scorer
3. Retrain LightGBM using trades from the winning strategy as labels
4. Route instruments to their best-performing strategy (e.g. trend for forex, meanrev for commodities)

Data lives in per-strategy SQLite DBs: stockio_trend.db, stockio_meanrev.db, stockio_momentum.db, stockio_llm.db

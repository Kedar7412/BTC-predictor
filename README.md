# polybot: Polymarket BTC 5-minute Up/Down bot

An autonomous bot for Polymarket's "Bitcoin Up or Down" 5-minute markets. It only
buys when its own probability estimate beats the ask price by a configurable
margin (positive edge and expected value) and the model is confident.
**Paper mode is the default. No profit is guaranteed. You can lose money.**

## Assumptions and the "jev like architecture" interpretation

"jev like architecture" was not defined in the request. I interpreted it as a
modular, event-driven pipeline of small single-purpose components wired by one
orchestrator loop. If you meant a specific project, the modules are easy to swap.

```
 PriceFeed ──► Signal (P_up) ──► Edge/EV ──► Risk + Kelly ──► Executor (paper|live)
 (feeds.py)   (signal.py)       (edge.py)    (risk.py)        (execution.py)
      ▲            ▲                 ▲            │                  │
 MarketProvider (market.py: Gamma discovery + CLOB books)            ▼
                          Engine loop (engine.py) ──► Monitor (JSONL, state.json, stats)
```

Other assumptions: market slug is `btc-updown-5m-<window_start_unix>`; the "Up"
outcome wins if BTC at window end is above BTC at window start (Binance spot
as proxy for the resolution source, which may differ slightly); live settlement
is approximated by price outcome, while real redemption happens on-chain.
Fee model is a conservative estimate (`fee_rate * p * (1-p)`), tune it to the current Polymarket schedule.

## The math

- Probability: `x = ln(S/K)`, `z = x / (sigma * sqrt(T)) + w*momentum`, `P(up) = Phi(z)`, clamped to [0.01, 0.99].
  K is the window open price, T seconds left, sigma realized vol per sqrt(second) (shrunk to a default when samples are few).
- Effective price: `eff = ask + slippage + fee`.
- Edge: `p - eff`. EV per share: `p*(1-c) - (1-p)*c - fee`, ROI: `EV/eff`.
- Conviction: model probability of the favored side.
- Kelly: `f* = (p - eff)/(1 - eff)`, stake = `kelly_fraction * f* * bankroll`, capped by max stake, bankroll and ask liquidity.

A trade requires ALL of: conviction >= `min_conviction`, edge >= `min_edge`, EV >= `min_ev`, ROI >= `min_roi`,
spread <= `max_spread`, enough liquidity, outside the no-trade windows at open and close,
one position per window, and the daily loss kill switch not tripped.

## Setup

```
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'        # add '.[live]' for py-clob-client
pytest -q
python -m polybot --simulate --max-cycles 5      # offline, no network
python -m polybot --dry-run                      # real data, paper fills
```

Edit `config.yaml` for thresholds. Secrets come only from environment variables (see `.env.example`).

## Live trading

Requires both `--live` and `CONFIRM_LIVE=yes`, plus `POLYMARKET_PRIVATE_KEY` (and optionally
`POLYMARKET_FUNDER`, API creds) and `pip install -e '.[live]'`. Without both, the bot refuses to start live.
Start tiny, verify in paper mode first, and treat the edge model as unproven: backtest it before risking funds.

## Outputs

`logs/trades.jsonl` (decisions, opens, settlements), `state.json` (bankroll, daily PnL for restart), periodic stats (win rate, PnL, average edge, skip reasons).

## Disclaimer

Not financial advice. Markets may be efficient, fees and latency can erase edge, and API behavior may change.

## Known limitations

- `market_shrink` blends the market mid into the probability used for the EV gate, a guard against model error. The model is not calibrated on historical books.
- Live mode books PnL from exchange prices, not the wallet or Polymarket's resolution feed, and does not redeem tokens. Start with a small bankroll.
- A window is marked traded before an order is posted, so an errored or timed-out live order is never retried in that window (it is recorded in `LiveExecutor.unreconciled`).
- Windows with no market or no open price are skipped; the bot never uses the current price as the window open.
- Backtests use simulated Polymarket prices and 1m closes (live polls every 5 s). They cannot prove an edge; see "Backtesting for 5 years".

## Backtesting for 5 years

Polymarket 5 minute markets have no 5 year history, so the backtest replays **Binance BTCUSDT 1m klines**
(data.binance.vision) through the real `signal.estimate`, `edge.best_opportunity`, `RiskManager.decide`
and `PaperExecutor` code. The market bid/ask are an **assumed model** (`backtest/market_model.py`), so results
show behaviour under stated assumptions and do **not** prove a real edge.

### Commands for your PC

```
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e '.[dev,backtest]'                          # backtest extra = matplotlib for PNG charts (optional)
python -m polybot.backtest --download-only --years 5 --data-dir ./bt_data     # resumable, ~60 monthly files
python -m polybot.backtest --suite quick --years 5 --data-dir ./bt_data --out ./bt_quick --workers 4
python -m polybot.backtest --suite full  --years 5 --data-dir ./bt_data --out ./bt_full  --workers 4 --resume
python -m polybot.backtest --synthetic --suite quick --years 0.5 --out ./bt_smoke   # offline smoke test
```

* `--suite quick` uses reduced grids (fewer scenarios x costs, 8 configs, 300 Monte Carlo paths).
  `--suite full` runs everything (9 scenarios, 3x3 cost grid on 3 scenarios, 40-config sweep with
  walk-forward, 2000 Monte Carlo paths, stress tests on the last 365 days). Without `--suite` you get one single run.
* `--workers N` parallelises signals and the many simulations (results are identical to 1 worker).
  Each worker holds a copy of the signal arrays, about 200 MB for 5 years, so 4 workers need roughly 1 GB.
* `--resume` reuses the signals cached in `--data-dir` as `signals_*.npz`.
  Signals are written after the first run, so a repeat run (or the full suite after the quick one) skips that step.
* Disk and time (rough, normal PC): 5 years is about 2.6M 1m rows, about 150 MB cached; ~1.5M 5 minute windows.
  Download takes minutes. Signals take roughly 10 to 25 minutes single process (faster with workers).
  One simulation over 5 years takes about 20 to 40 s. The quick suite is typically 15 to 40 minutes and the full
  suite 1 to 2 hours with 4 workers. A `--synthetic --years 0.5` quick suite takes about 1.5 minutes.
  Progress is printed to stderr per stage.
* Outputs in `--out`: `report.md`, `report.html`, `results.json`, CSVs (`trades`, `equity`, `scenarios`,
  `cost_grid`, `controls`, `sweep`, `walkforward`, `montecarlo`, `stress`, `calibration`) and PNG charts when matplotlib is installed
  (otherwise ASCII charts in the report).

### Reading the report

The report opens with the assumptions and limitations. Then:

1. **All scenarios side by side.** `efficient` (market equals the bot's probability, only costs) comes first, then a
   moderately noisy market, then the rest. There is no single headline number. A scenario is flagged
   `WARNING artifact` when Sharpe is above 5 or the return is above 100x. A stale or lagged market is
   exploitable by construction (the "market" literally lags the price the bot sees), so a stale scenario that turns 1000 into
   hundreds of thousands says something about the assumption, not about the strategy.
2. **No-information controls.** The bot runs on a market whose price is the true probability (outcomes drawn from
   the price) and on an independent random walk. It must lose money to costs. If a control makes money, the
   cost accounting is wrong. This validates cost accounting only: real outcomes never enter the control, so it cannot
   detect look-ahead leaks (the no-look-ahead tests cover those).
3. **Cost sweep and break-even.** PnL against half spread and fee rate, and the smallest market noise or
   lag at which the strategy still profits.
4. **Calibration (Brier).** 0.25 is a coin flip. A Brier below 0.25 means the bot's probability tracks BTC moves; it says
   nothing about Polymarket prices.
5. **Walk-forward and held-out split.** Parameters are chosen on train segments only and scored frozen on the next
   test segment. Look at the test/train degradation ratio, whether the neighbourhood is a plateau or a spike, the
   deflated-Sharpe style haircut for the number of trials, and the default parameters out of sample. The sweep
   tunes against the assumed market, and winners are never written to `config.yaml`.
6. **Monte Carlo.** Block bootstrap and shuffle of the trade PnL: spread of final return and drawdown, probability of
   losing half, of drawdown above 30%, and of hitting the daily loss kill.
7. **Stress tests.** Volatility spikes, data gaps, flash gaps, higher fees and slippage, adverse selection (ask +0.5 to 3c)
   and a combined worst case, flagged if the result turns negative or drawdown exceeds 30%.

Known optimism: noise in the `noisy` scenarios is drawn independently per decision tick and the replay takes the first
tick that passes the gates, so the bot gets up to four independent mispricing draws per window. Real mispricing persists
across a window, so treat noisy results (and the sweep and stress tests that use them) as upper bounds.

### Why simulated prices cannot prove edge

Every bid and ask in the backtest is generated from the assumed model. If the model says the market is
sloppy, the bot will find edges; if it says the market is sharp, it will not. Both are assumptions. The
only evidence of a real edge is real Polymarket books, so record them (below) and rerun.

**Signals use 1m closes.** Decisions are taken at elapsed 60, 120, 180 and 240 s on the last completed 1m close.
The live bot polls every 5 s with intra-minute prices, so entry timing, fills and risk checks differ.
Treat the backtest as an approximation of the live cadence.

### Recording real books

```
python -m polybot.record --out books.jsonl            # read only, no orders, no keys; leave it running for weeks
python -m polybot.backtest --suite quick --years 1 --books books.jsonl --books-mode only    # REAL books only
python -m polybot.backtest --suite quick --years 1 --books books.jsonl --books-mode mixed   # REAL where recorded, MODEL elsewhere
```

The reference run then uses recorded books for windows that have them and the report tags REAL vs MODEL tick counts.
The scenario matrix still uses the model.

### Go / no-go checklist before any live money

- [ ] Controls pass (costs lose money with no information).
- [ ] Positive after the cost sweep at realistic spread and fee, not only in the most favourable cell.
- [ ] Walk-forward out-of-sample is positive in most folds, with test/train degradation near or above 0.5 and a plateau, not a spike.
- [ ] No result in the final decision depends on a flagged `WARNING artifact` scenario.
- [ ] Monte Carlo drawdown p95 and P(lose half) are acceptable for your bankroll.
- [ ] No stress turns the result negative, or you accept that case.
- [ ] Weeks of recorded real books replayed with `--books`, edge still present after spread and fees.
- [ ] Weeks of paper trading in the live loop. Then start with a tiny stake.


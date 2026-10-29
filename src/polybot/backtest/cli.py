"""python -m polybot.backtest : single backtest run."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

from ..config import Settings, load_settings
from . import data as data_mod
from .market_model import ASSUMPTIONS_BANNER, SCENARIOS, AssumedMarket
from .metrics import calibration, compute_metrics
from .scenarios import implausible
from .sim import TRADE_COLS, build_signals, run_sim


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m polybot.backtest",
                                description="Backtest polybot on Binance BTCUSDT 1m klines with an ASSUMED market model")
    p.add_argument("--years", type=float, default=5.0, help="history length (default 5)")
    p.add_argument("--start", help="YYYY-MM-DD (overrides --years)")
    p.add_argument("--end", help="YYYY-MM-DD inclusive (default: yesterday UTC)")
    p.add_argument("--data-dir", default="./bt_data", help="cache for downloaded klines")
    p.add_argument("--out", default="./bt_out", help="output directory")
    p.add_argument("--config", default="config.yaml", help="polybot settings yaml")
    p.add_argument("--scenario", choices=SCENARIOS, default="noisy",
                   help="assumed market for the single run (default: moderately noisy). Stale scenarios are "
                        "exploitable by construction and produce absurd returns on purpose")
    p.add_argument("--noise-sd", type=float, default=0.3, help="logit noise of the assumed market")
    p.add_argument("--lag-minutes", type=int, default=2, help="staleness for lagged scenarios")
    p.add_argument("--half-spread", type=float, default=0.01)
    p.add_argument("--ask-size-usd", type=float, default=200.0)
    p.add_argument("--adverse-selection", type=float, default=0.0, help="extra ask shift (latency model)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bankroll", type=float)
    p.add_argument("--workers", type=int, default=1, help="processes for signal computation (results identical)")
    p.add_argument("--synthetic", action="store_true", help="offline fake data (no edge, for smoke tests)")
    p.add_argument("--csv", nargs="+", help="local Binance-format kline csv/zip files instead of downloading")
    p.add_argument("--download-only", action="store_true", help="only download/cache data, then exit")
    p.add_argument("--suite", choices=("none", "quick", "full"), default="none",
                   help="run the hard-test suite (scenario matrix, controls, walk-forward, sweep, Monte Carlo, "
                        "stress) and write report.md. 'none' = single run")
    p.add_argument("--resume", action="store_true", help="reuse cached signals (.npz) from --data-dir")
    p.add_argument("--mc-paths", type=int, help="Monte Carlo paths (default 2000 full, 300 quick)")
    p.add_argument("--stress-days", type=float, help="days of data for data stresses (default 365 full, all quick)")
    p.add_argument("--books", help="JSONL of REAL recorded books (python -m polybot.record)")
    p.add_argument("--books-mode", choices=("only", "mixed"), default="only",
                   help="only: skip windows without recordings; mixed: model them")
    p.add_argument("--no-verify", action="store_true", help="skip checksum verification of downloads")
    return p


def _dates(a) -> tuple[dt.date, dt.date]:
    end = dt.date.fromisoformat(a.end) if a.end else dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)
    start = dt.date.fromisoformat(a.start) if a.start else end - dt.timedelta(days=int(round(a.years * 365.25)))
    if start > end:
        raise SystemExit("--start is after --end")
    return start, end


def load_data(a) -> data_mod.Klines:
    if a.synthetic:
        return data_mod.synthetic(a.years, a.seed)
    if a.csv:
        return data_mod.load_csv(a.csv)
    start, end = _dates(a)
    return data_mod.load_binance(start, end, a.data_dir, verify=not a.no_verify)


def make_source(a):
    mk = AssumedMarket(a.scenario, a.noise_sd, a.lag_minutes, a.half_spread, a.ask_size_usd,
                       a.adverse_selection, a.seed)
    if a.books:
        from .books import RecordedBooks
        return RecordedBooks(a.books, a.books_mode, mk if a.books_mode == "mixed" else None)
    return mk


def write_outputs(out: Path, tr: dict, summary: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    n = len(tr["pnl"])
    with open(out / "trades.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(TRADE_COLS)
        for i in range(n):
            w.writerow([int(tr[c][i]) if tr[c].dtype.kind in "iub" else f"{tr[c][i]:.6f}" for c in TRADE_COLS])
    with open(out / "equity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "bankroll"])
        w.writerow([summary["data"].get("first", 0), summary["metrics"]["start_bankroll"]])
        for i in range(n):
            w.writerow([int(tr["settle_ts"][i]), f"{tr['bankroll'][i]:.4f}"])
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))


def run(a, s: Settings | None = None) -> dict:
    s = s or load_settings(a.config)
    if a.bankroll is not None:
        s.bankroll = a.bankroll
    print(ASSUMPTIONS_BANNER, file=sys.stderr)
    t0 = time.time()
    k = load_data(a)
    q = k.quality
    print(f"data: {q.get('rows')} minutes, missing {q.get('missing_minutes')}, gaps {q.get('gap_count')}, "
          f"duplicates {q.get('duplicates')}, outliers {q.get('outliers_gt_5pct_1m')}", file=sys.stderr)
    if a.suite != "none":
        from .suite import run_suite
        src = make_source(a) if a.books else None
        R = run_suite(a, s, k, src)
        print_suite_summary(R)
        return R
    sigs = build_signals(k, s, workers=a.workers, progress=True)
    src = make_source(a)
    res = run_sim(sigs, s, src, progress=True)
    tr = res.arrays()
    start_ts, end_ts = int(sigs.ws[0]), int(sigs.ws[-1]) + 300
    metrics = compute_metrics(tr, s.bankroll, start_ts, end_ts)
    cal = calibration(sigs.prob, sigs.valid, sigs.up_won)
    nw = sigs.stats["windows_total"]
    total_s = sigs.stats["signal_seconds"] + res.seconds
    summary = {
        "WARNING": "Polymarket bid/ask are MODELLED, not historical. This does not prove a real edge.",
        "assumed_market": src.describe(), "settings": {k_: v for k_, v in vars(s).items()},
        "data": q, "windows": sigs.stats, "sim": res.counters, "metrics": metrics, "calibration": cal,
        "throughput": {"windows": nw, "seconds": total_s, "windows_per_sec": nw / total_s if total_s else None,
                       "est_minutes_for_1.5M_windows": 1.5e6 / (nw / total_s) / 60 if total_s and nw else None},
        "wall_seconds": time.time() - t0,
    }
    write_outputs(Path(a.out), tr, summary)
    print_summary(summary, src)
    return summary


def print_suite_summary(R: dict) -> None:
    print(ASSUMPTIONS_BANNER)
    print("Scenarios (efficient and moderately noisy first, no single headline):")
    for n, r in R["scenarios"]["baseline"].items():
        m = r["metrics"]
        print(f"  {n:26s} trades {m['trades']:7d}  return {100 * m['total_return']:12.1f}%  "
              f"Sharpe {m['sharpe'] if m['sharpe'] is None else round(m['sharpe'], 2)}"
              f"{'  [WARNING artifact of assumed market]' if r['warning'] else ''}")
    print(R["controls"]["verdict"])
    print("Live bot decides every 5 s, this replay uses 1m closes: results are approximate.")
    print(f"Report: {R['meta']['out']}/report.md")


def print_summary(sm: dict, src: AssumedMarket) -> None:
    m, c = sm["metrics"], sm["calibration"]
    f = lambda v, p=3: "n/a" if v is None else f"{v:.{p}f}"
    print(ASSUMPTIONS_BANNER)
    print(f"market model: {src.describe()}")
    print(f"windows: {sm['windows']['windows_valid']} valid of {sm['windows']['windows_total']} "
          f"({sm['windows']['windows_skipped_missing_minutes']} skipped for missing minutes)")
    print(f"trades {m['trades']}  win rate {f(m['win_rate'])}  avg edge {f(m['avg_edge'])}  "
          f"trades/day {f(m['trades_per_day'], 2)}")
    print(f"bankroll {m['start_bankroll']:.2f} -> {m['final_bankroll']:.2f}  return {100 * m['total_return']:.1f}%  "
          f"CAGR {100 * m['cagr']:.1f}%")
    print(f"max drawdown {100 * m['max_drawdown']:.1f}% ({f(m['max_drawdown_duration_days'], 1)} days)  "
          f"Sharpe {f(m['sharpe'], 2)}  Sortino {f(m['sortino'], 2)}  profit factor {f(m['profit_factor'], 2)}")
    print(f"calibration: Brier {f(c['brier'], 4)} (0.25 = coin flip), skill vs 0.5 {f(c['brier_skill_vs_0.5'], 4)}")
    print(f"throughput {sm['throughput']['windows_per_sec']:,.0f} windows/s; outputs in the --out directory")
    w = implausible(m)
    if w:
        print(w)
    print("NOTE: signals use 1m closes at 60/120/180/240 s; the live bot polls every 5 s, so live behaviour differs.")
    print("REMINDER: modelled market prices. Not evidence of a tradable edge. Run --suite quick for all scenarios.")


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.download_only:
        if a.synthetic or a.csv:
            raise SystemExit("--download-only cannot be combined with --synthetic/--csv")
        start, end = _dates(a)
        files = data_mod.ensure_data(start, end, a.data_dir, verify=not a.no_verify)
        print(f"cached {len(files)} files in {a.data_dir}")
        return 0
    run(a)
    return 0

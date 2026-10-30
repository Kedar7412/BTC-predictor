"""Write report.md / report.html / results.json and CSVs. The report opens with the assumptions banner."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import numpy as np

from .market_model import ASSUMPTIONS_BANNER
from .scenarios import MAX_PLAUSIBLE_MULTIPLE, MAX_PLAUSIBLE_SHARPE

LIMITATIONS = [
    "There is no Polymarket history for 5 years. The 5 minute BTC markets are new. Every book price "
    "(bid, ask, size) in this report is SIMULATED by an assumed market model unless a section says REAL books.",
    "Because prices are simulated, these results CANNOT prove a real edge. They show how the strategy behaves "
    "if the market is as inefficient as the stated scenario. Different assumption, different answer.",
    "The 'efficient' scenario (market equals the bot's own probability) and the 'noisy' scenarios are centred "
    "on the bot's model and are optimistic upper bounds. Lagged/stale scenarios are exploitable by "
    "construction: a very high result there is an artifact of the assumed staleness, not a finding.",
    "In the noisy scenarios the noise is drawn independently at each of the four decision ticks, and the "
    "simulator takes the first tick that passes the gates. The bot therefore gets up to four independent "
    "mispricing draws per window, whereas real mispricing persists through a window. This inflates trade count "
    "and win rate, so noisy results are an optimistic upper bound. The reference numbers also assume zero latency.",
    "Prices come from Binance BTCUSDT 1m closes. Polymarket resolves on a different source (Chainlink), so "
    "outcomes near the open price can differ from what is simulated here.",
    "Resolution is 1 minute. Signals are computed on 1m closes at elapsed 60, 120, 180 and 240 s. The live bot "
    "polls every 5 s and sees intra-minute prices, so its decisions, fills and risk checks differ from this "
    "replay. Treat the backtest as an approximation of the live cadence, not a replica.",
    "Survivorship and regime: one asset, one 5 year path, a few volatility regimes. Past regimes may not repeat.",
    "Parameter sweeps tune the bot to the assumed market, not to Polymarket. Winners are never written to config.yaml.",
    "Fills are assumed at the quoted ask with limited size; real queues, rejections and latency are not modelled "
    "except through the optional adverse_selection shift.",
]


def pct(v, d=1):
    return "n/a" if v is None else f"{100 * v + 0.0:.{d}f}%".replace("-0.0%", "0.0%")


def num(v, d=2):
    if v is None:
        return "n/a"
    if isinstance(v, float) and v == float("-inf"):
        return "-inf"
    return f"{v:,.{d}f}"


def md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def ascii_chart(values: np.ndarray, width: int = 70, height: int = 10, label: str = "") -> str:
    if len(values) < 2:
        return "(no data)"
    idx = np.linspace(0, len(values) - 1, width).astype(int)
    v = values[idx]
    lo, hi = float(v.min()), float(v.max())
    span = (hi - lo) or 1.0
    rows = [[" "] * width for _ in range(height)]
    for x, y in enumerate(v):
        rows[height - 1 - int((y - lo) / span * (height - 1))][x] = "*"
    lines = ["".join(r) for r in rows]
    return f"{label} (min {lo:,.2f}, max {hi:,.2f})\n" + "\n".join(lines)


def charts(out: Path, equity: np.ndarray, times: np.ndarray) -> tuple[list[str], str]:
    """PNG charts if matplotlib is importable, else ASCII text. Returns (png names, ascii text)."""
    if len(equity) < 2:
        return [], "(no trades)"
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return [], ascii_chart(equity, label="equity") + "\n\n" + ascii_chart(dd, label="drawdown")
    step = max(1, len(equity) // 5000)
    t = times[::step].astype("datetime64[s]")
    names = []
    for name, y, title in (("equity.png", equity[::step], "Equity (log scale)"),
                           ("drawdown.png", dd[::step], "Drawdown")):
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot(t, y)
        if name == "equity.png" and equity.min() > 0:
            ax.set_yscale("log")
        ax.set_title(title + " (MODELLED market)")
        fig.tight_layout()
        fig.savefig(out / name, dpi=110)
        plt.close(fig)
        names.append(name)
    return names, ""


def build_markdown(R: dict, png: list[str], ascii_txt: str) -> str:
    L: list[str] = []
    A = L.append
    A("# ASSUMPTIONS AND LIMITATIONS (READ FIRST)")
    A("")
    A("```")
    A(ASSUMPTIONS_BANNER)
    A("```")
    A("")
    for t in LIMITATIONS:
        A(f"- {t}")
    A("")
    A(f"# polybot backtest report ({R['meta']['suite']} suite)")
    A("")
    A(f"Period: {R['meta'].get('period', 'n/a')}. Windows: {R['windows'].get('windows_valid')} valid of "
      f"{R['windows'].get('windows_total')}. Data quality: {json.dumps(R.get('data', {}), default=str)}")
    if R.get("books"):
        A("")
        A(f"**Book source: {R['books'].get('source')}** (REAL vs MODEL counts: "
          f"{R['books'].get('real_book_ticks')} real ticks, {R['books'].get('model_book_ticks')} modelled ticks)")
    A("")
    A("## 1. All scenarios side by side")
    A("")
    A("Order: `efficient` (no information edge, only costs) first, then a moderately noisy market, then the "
      "rest. No single scenario is a headline. Compare them. Rows flagged WARNING are artifacts of the "
      f"assumed market (Sharpe > {MAX_PLAUSIBLE_SHARPE:g} or more than {MAX_PLAUSIBLE_MULTIPLE:g}x return).")
    A("")
    rows, warns = [], []
    for n, r in R["scenarios"]["baseline"].items():
        m = r["metrics"]
        flag = "WARNING artifact" if r["warning"] else ""
        rows.append([n, m["trades"], pct(m["win_rate"]), pct(m["total_return"]), pct(m["cagr"]),
                     num(m["sharpe"]), pct(m["max_drawdown"]), flag])
        if r["warning"]:
            warns.append(f"- **{n}**: {r['warning']}")
    A(md_table(["scenario", "trades", "win rate", "total return", "CAGR", "Sharpe", "max DD", "flag"], rows))
    if warns:
        A("")
        A("\n".join(warns))
    A("")
    A("## 2. No-information controls (is the harness printing free money?)")
    A("")
    c = R["controls"]
    A(md_table(["control", "trades", "total return", "win rate"],
               [[k, v["trades"], pct(v["total_return"]), pct(v["win_rate"])] for k, v in c.items()
                if k != "verdict"]))
    A("")
    A(f"**{c['verdict']}**")
    A("")
    A("`efficient`: market equals the bot probability, so the bot should not trade. `no_info_noisy_*`: the market "
      "is noisy around the bot probability but outcomes are drawn from the market price, so the bot has "
      "apparent edges that carry no information. It trades and pays spread, fee and slippage, so PnL must be "
      "negative. This validates cost accounting only: real outcomes never enter the control, so it cannot detect look-ahead leaks (the no-look-ahead tests cover those). Repeated on an independent random-walk series.")
    A("")
    A("## 3. Cost sweep (half spread x fee rate)")
    A("")
    for sc, cells in R["scenarios"]["cost_grid"].items():
        A(f"**{sc}**: total return")
        A("")
        rows = [[f"{v['half_spread']}", f"{v['fee_rate']}", v["trades"], pct(v["total_return"]),
                 "WARNING artifact" if v["warning"] else ""] for v in cells.values()]
        A(md_table(["half spread", "fee rate", "trades", "total return", "flag"], rows))
        A("")
    A("## 4. Market-price sensitivity and break-even")
    A("")
    se = R["scenarios"]["sensitivity"]
    A(md_table(["noise sd", "trades", "total return", "Sharpe"],
               [[r["noise_sd"], r["trades"], pct(r["total_return"]), num(r["sharpe"])] for r in se["vs_noise_sd"]]))
    A("")
    A(md_table(["lag minutes", "trades", "total return", "Sharpe"],
               [[r["lag_minutes"], r["trades"], pct(r["total_return"]), num(r["sharpe"])]
                for r in se["vs_lag_minutes"]]))
    A("")
    be = R["scenarios"]["break_even"]
    A(f"Smallest noise sd with positive return: {be['min_noise_sd_with_positive_return']}. "
      f"Smallest lag (min) with positive return: {be['min_lag_minutes_with_positive_return']}. {be['note_noise']}")
    A("")
    A(md_table(["scenario", "max half spread > 0", "max fee rate > 0", "profitable cells"],
               [[k, v["max_half_spread_positive"], v["max_fee_rate_positive"], v["cells_positive"]]
                for k, v in be["costs"].items()]))
    A("")
    A("## 5. Reference run")
    A("")
    b = R["baseline"]
    m = b["metrics"]
    A(f"Market: {json.dumps(b['market'])}")
    A("")
    A(f"Trades {m['trades']}, win rate {pct(m['win_rate'])}, bankroll {num(m['start_bankroll'])} -> "
      f"{num(m['final_bankroll'])}, CAGR {pct(m['cagr'])}, Sharpe {num(m['sharpe'])}, Sortino {num(m['sortino'])}, "
      f"max DD {pct(m['max_drawdown'])}, profit factor {num(m['profit_factor'])}.")
    if b.get("warning"):
        A("")
        A(f"**{b['warning']}**")
    A("")
    for p in png:
        A(f"![{p}]({p})")
    if ascii_txt:
        A("```")
        A(ascii_txt)
        A("```")
    cal = b["calibration"]
    A("")
    A(f"Calibration of the bot probability over all windows: Brier {num(cal['brier'], 4)} "
      f"(0.25 = coin flip), skill vs 0.5 {num(cal['brier_skill_vs_0.5'], 4)}. Brier below 0.25 only means the model "
      "tracks BTC moves. It says nothing about Polymarket prices.")
    A("")
    A(md_table(["bin", "n", "mean predicted", "freq up"],
               [[f"{r['lo']:.1f}-{r['hi']:.1f}", r["n"], num(r["mean_pred"], 3), num(r["freq_up"], 3)]
                for r in cal["reliability"]]))
    A("")
    if "walkforward" in R:
        w = R["walkforward"]
        A("## 6. Walk-forward and held-out evaluation")
        A("")
        A(f"{w['NOTE']} Segments: {w['segments']} x {w['segment_rows']} windows, train {w['train_segments']} / "
          f"test {w['test_segments']} segments, {w['n_trials_incl_neighbours']} configs tried.")
        A("")
        rows = []
        for i, f in enumerate(w["folds"]):
            if not f.get("config"):
                rows.append([i, f["train_segments"], f["test_segments"], f.get("note"), "", "", "", ""])
                continue
            rows.append([i, f"{f['train_segments'][0]}-{f['train_segments'][-1]}", f["test_segments"],
                         num(f["train_score"], 4), num(f["test_score"], 4), f["test_trades"],
                         num(f["degradation_test_over_train"]), f["stability"]["verdict"]])
        A(md_table(["fold", "train segs", "test segs", "train score", "test score", "test trades",
                    "test/train", "neighbourhood"], rows))
        A("")
        st = w["stitched_oos"]
        mult = st["oos_multiple"]
        big = mult is not None and mult > 100
        A(f"Stitched out-of-sample: {st['folds_with_a_winner']} folds, {st['oos_trades']} trades, "
          f"multiple {f'{mult:.3e}' if big else num(mult, 3)} (segments compounded back to back), "
          f"positive folds {st['oos_positive_folds']}, "
          f"mean test/train degradation {num(st['mean_degradation_test_over_train'])}. "
          f"Default params OOS log return sum {num(st['default_params_oos_log_return_sum'], 4)}.")
        if big:
            A("")
            A("**WARNING artifact: the stitched multiple is above 100x. It compounds every test segment against "
              "the ASSUMED market, so it measures how exploitable that model is, not a real edge.**")
        A("")
        sp = w.get("fixed_split")
        if sp and sp.get("config"):
            A(f"Fixed split (last 30% never used for tuning): train {num(sp['train_score'], 4)}, held-out "
              f"{num(sp['test_score'], 4)}, trades {sp['test_trades']}, degradation "
              f"{num(sp['degradation_test_over_train'])}, stability: {sp['stability']['verdict']}. "
              f"Deflation: {json.dumps(sp['deflation'])}. Winner: {json.dumps(sp['config'])}")
            A("")
        A("### Sweep (top configs by in-sample score; in-sample numbers are optimistic by construction)")
        A("")
        A(md_table(["config", "IS score", "IS trades", "held-out score"],
                   [[json.dumps(r["config"]), num(r["is_score"], 4), r["is_trades"], num(r["oos_score"], 4)]
                    for r in w["sweep_table_by_in_sample_score"]]))
        A("")
    if "montecarlo" in R:
        mc = R["montecarlo"]
        A("## 7. Monte Carlo (resampling the trade PnL)")
        A("")
        if mc.get("n_paths"):
            rows = []
            for mode in ("block_bootstrap", "shuffle"):
                x = mc[mode]
                rows.append([mode, pct(x["final_return"]["p5"]), pct(x["final_return"]["p50"]),
                             pct(x["final_return"]["p95"]), pct(x["max_drawdown"]["p5"]),
                             pct(x["max_drawdown"]["p50"]), pct(x["max_drawdown"]["p95"]),
                             pct(x["prob_loss"]), pct(x["prob_lose_half"]), pct(x["prob_drawdown_gt_30pct"])])
            A(md_table(["mode", "ret p5", "ret p50", "ret p95", "DD p5", "DD p50", "DD p95", "P(loss)",
                        "P(lose 50%)", "P(DD>30%)"], rows))
            dk = mc.get("daily_loss_kill")
            if dk:
                A("")
                A(f"Daily loss kill ({dk['limit']}): P(any day hits it) = {pct(dk['prob_any_day_hits_limit'])}, "
                  f"observed days at the limit {dk['observed_days_at_or_below_limit']} of {dk['days']}.")
        else:
            A(mc.get("note", "n/a"))
        A("")
    if "stress" in R:
        st = R["stress"]
        A("## 8. Stress tests")
        A("")
        A(f"Period {st['period_days']:.0f} days, reference market. {st['note']}")
        A("")
        A(md_table(["stress", "trades", "total return", "max DD", "flags"],
                   [[n, v["trades"], pct(v["total_return"]), pct(v["max_drawdown"]), "; ".join(v["flags"])]
                    for n, v in st["results"].items()]))
        A("")
        A("Strategy flagged in at least one stress." if st["any_flag"] else "No stress flagged.")
        A("")
    A("## Runtime")
    A("")
    A(f"Wall time {R['meta'].get('wall_seconds', 0):.0f} s, workers {R['meta'].get('workers')}.")
    return "\n".join(L) + "\n"


def to_html(md: str) -> str:
    out = ["<html><head><meta charset='utf-8'><title>polybot backtest</title>"
           "<style>body{font-family:sans-serif;max-width:1100px;margin:2em auto}table{border-collapse:collapse}"
           "td,th{border:1px solid #999;padding:2px 6px;font-size:13px}pre{background:#eee;padding:6px}"
           "</style></head><body>"]
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            j = i + 1
            blk = []
            while j < len(lines) and not lines[j].startswith("```"):
                blk.append(lines[j])
                j += 1
            out.append("<pre>" + html.escape("\n".join(blk)) + "</pre>")
            i = j + 1
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            out.append("<table>")
            for ri, r in enumerate(rows):
                if ri == 1:
                    continue
                tag = "th" if ri == 0 else "td"
                out.append("<tr>" + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in r) + "</tr>")
            out.append("</table>")
        elif ln.startswith("!["):
            out.append(f"<img src='{html.escape(ln[ln.index('(') + 1:-1])}'>")
            i += 1
        elif ln.startswith("#"):
            lvl = len(ln) - len(ln.lstrip("#"))
            out.append(f"<h{lvl}>{html.escape(ln.lstrip('# '))}</h{lvl}>")
            i += 1
        elif ln.startswith("- "):
            out.append("<ul><li>" + html.escape(ln[2:]) + "</li></ul>")
            i += 1
        elif ln.strip():
            out.append("<p>" + html.escape(ln.replace("**", "")) + "</p>")
            i += 1
        else:
            i += 1
    out.append("</body></html>")
    return "\n".join(out)


def _csv(path: Path, headers: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


def write_csvs(out: Path, R: dict) -> None:
    sc = R["scenarios"]
    _csv(out / "scenarios.csv", ["scenario", "trades", "win_rate", "total_return", "cagr", "sharpe",
                                 "max_drawdown", "warning"],
         [[n, r["metrics"]["trades"], r["metrics"]["win_rate"], r["metrics"]["total_return"], r["metrics"]["cagr"],
           r["metrics"]["sharpe"], r["metrics"]["max_drawdown"], bool(r["warning"])]
          for n, r in sc["baseline"].items()])
    _csv(out / "cost_grid.csv", ["scenario", "half_spread", "fee_rate", "trades", "total_return", "sharpe"],
         [[s_, v["half_spread"], v["fee_rate"], v["trades"], v["total_return"], v["sharpe"]]
          for s_, cells in sc["cost_grid"].items() for v in cells.values()])
    _csv(out / "controls.csv", ["control", "trades", "total_return", "win_rate"],
         [[k, v["trades"], v["total_return"], v["win_rate"]] for k, v in R["controls"].items() if k != "verdict"])
    cal = R["baseline"]["calibration"]["reliability"]
    _csv(out / "calibration.csv", ["lo", "hi", "n", "mean_pred", "freq_up"],
         [[r["lo"], r["hi"], r["n"], r["mean_pred"], r["freq_up"]] for r in cal])
    if "walkforward" in R:
        w = R["walkforward"]
        _csv(out / "sweep.csv", ["config", "is_score", "is_trades", "heldout_score"],
             [[json.dumps(r["config"]), r["is_score"], r["is_trades"], r["oos_score"]]
              for r in w["sweep_table_by_in_sample_score"]])
        _csv(out / "walkforward.csv", ["fold", "train_segments", "test_segments", "config", "train_score",
                                       "test_score", "test_trades", "degradation", "stability"],
             [[i, f["train_segments"], f["test_segments"], json.dumps(f.get("config")), f.get("train_score"),
               f.get("test_score"), f.get("test_trades"), f.get("degradation_test_over_train"),
               (f.get("stability") or {}).get("verdict")] for i, f in enumerate(w["folds"])])
    if "montecarlo" in R and R["montecarlo"].get("n_paths"):
        mc = R["montecarlo"]
        _csv(out / "montecarlo.csv", ["mode", "ret_p5", "ret_p50", "ret_p95", "dd_p5", "dd_p50", "dd_p95",
                                      "prob_loss", "prob_lose_half", "prob_dd_gt_30"],
             [[m_, mc[m_]["final_return"]["p5"], mc[m_]["final_return"]["p50"], mc[m_]["final_return"]["p95"],
               mc[m_]["max_drawdown"]["p5"], mc[m_]["max_drawdown"]["p50"], mc[m_]["max_drawdown"]["p95"],
               mc[m_]["prob_loss"], mc[m_]["prob_lose_half"], mc[m_]["prob_drawdown_gt_30pct"]]
              for m_ in ("block_bootstrap", "shuffle")])
    if "stress" in R:
        _csv(out / "stress.csv", ["stress", "trades", "total_return", "max_drawdown", "sharpe", "flags"],
             [[n, v["trades"], v["total_return"], v["max_drawdown"], v["sharpe"], "; ".join(v["flags"])]
              for n, v in R["stress"]["results"].items()])


def write_report(out: Path, R: dict, equity: np.ndarray, times: np.ndarray) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    png, ascii_txt = charts(out, equity, times)
    md = build_markdown(R, png, ascii_txt)
    (out / "report.md").write_text(md)
    (out / "report.html").write_text(to_html(md))
    (out / "results.json").write_text(json.dumps(R, indent=2, default=_json_default))
    write_csvs(out, R)
    return out / "report.md"


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

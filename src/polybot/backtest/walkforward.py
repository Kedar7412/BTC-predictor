"""Walk-forward and fixed in-sample / out-of-sample evaluation on top of sweep.py.

Walk-forward: choose params on TRAIN segments (default 4 x 91 days = 12 months), evaluate the frozen
choice on the next TEST segment (default 1 x 91 days = 3 months), then step forward by one test
segment. Out-of-sample results are stitched. Train and test segments never overlap.
Fixed split: the last 30% of segments are held out and never used for tuning.
"""
from __future__ import annotations

import math

import numpy as np

from ..config import Settings
from . import sweep as sw


def folds(n_segs: int, train_segs: int, test_segs: int) -> list[tuple[list[int], list[int]]]:
    """Rolling windows of segment indices; train and test are disjoint and train precedes test."""
    out = []
    a = 0
    while a + train_segs + test_segs <= n_segs:
        out.append((list(range(a, a + train_segs)), list(range(a + train_segs, a + train_segs + test_segs))))
        a += test_segs
    return out


def split_is_oos(n_segs: int, oos_fraction: float = 0.3) -> tuple[list[int], list[int]]:
    n_oos = max(1, int(round(n_segs * oos_fraction)))
    n_oos = min(n_oos, n_segs - 1) if n_segs > 1 else 0
    return list(range(n_segs - n_oos)), list(range(n_segs - n_oos, n_segs))


def _row(cache, cfg, train, test, min_trades, fixed_mom, dfl) -> dict:
    res = cache[sw.key_of(cfg)]
    tr_s = sw.score(res, train, min_trades)
    te_s = sw.score(res, test, 0)
    deg = (te_s / tr_s) if (math.isfinite(tr_s) and tr_s > 0) else None
    return {"config": cfg, "train_score": None if not math.isfinite(tr_s) else tr_s, "test_score": te_s,
            "test_trades": int(sum(res[i]["trades"] for i in test)),
            "train_trades": int(sum(res[i]["trades"] for i in train)),
            "test_pnl": float(sum(res[i]["pnl"] for i in test)),
            "degradation_test_over_train": deg,
            "stability": sw.stability(cache, cfg, train, min_trades, fixed_mom), "deflation": dfl}


def run_walkforward(get_sigs, s: Settings, n_rows: int, workers: int = 1, full: bool = False,
                    train_segs: int | None = None, test_segs: int = 1, seed: int = 0) -> dict:
    cfgs = sw.sweep_settings(full)
    train_segs = train_segs or (4 if full else 2)
    segs = sw.segment_plan(n_rows, train_segs, test_segs)
    market = sw.market_for(cfgs["market_name"], seed)
    fixed_mom = cfgs["fixed_momentum"]
    min_trades = cfgs["min_trades"]
    configs = sw.sample_configs(s, cfgs["n_configs"], seed, fixed_mom)
    cache: dict = {}
    sw.evaluate(get_sigs, s, configs, segs, market, workers, cache)

    fl = folds(len(segs), train_segs, test_segs)
    is_idx, oos_idx = split_is_oos(len(segs))
    plan = [("fold", tr, te) for tr, te in fl] + ([("split", is_idx, oos_idx)] if oos_idx and is_idx else [])
    # neighbours of every winner, evaluated after selection (selection itself saw train segments only)
    winners = {}
    for kind, tr, te in plan:
        best, _ = sw.select_best(cache, configs, tr, min_trades)
        winners[(kind, tuple(tr))] = best
    # stability() only scores neighbours on their winner's train segments, so simulate just those
    extra, need = [], {}
    for (kind, tr), w in winners.items():
        if w:
            for n in sw.neighbours(w, fixed_mom):
                extra.append(n)
                need.setdefault(sw.key_of(n), set()).update(tr)
    sw.evaluate(get_sigs, s, extra, segs, market, workers, cache, "neighbours", only=need)
    n_trials = len(cache)

    base = sw.base_config(s)
    out_folds, split = [], None
    for kind, tr, te in plan:
        best = winners[(kind, tuple(tr))]
        if best is None:
            row = {"train_segments": tr, "test_segments": te, "config": None,
                   "note": f"no config reached {min_trades} train trades"}
        else:
            rets = sw.winner_returns(get_sigs, s, best, segs, tr, market)
            row = {"train_segments": tr, "test_segments": te,
                   **_row(cache, best, tr, te, min_trades, fixed_mom, sw.deflation(rets, n_trials))}
            row["frozen_default_test_score"] = sw.score(cache[sw.key_of(base)], te, 0)
        if kind == "fold":
            out_folds.append(row)
        else:
            split = row
    scored = [f for f in out_folds if f.get("config")]
    te_scores = [f["test_score"] for f in scored]
    base_te = [f["frozen_default_test_score"] for f in scored]
    stitched = {
        "folds": len(out_folds), "folds_with_a_winner": len(scored),
        "oos_trades": int(sum(f["test_trades"] for f in scored)),
        "oos_log_return_sum": float(np.sum(te_scores)) if te_scores else None,
        "oos_multiple": float(math.exp(sum(f["test_score"] * len(f["test_segments"]) for f in scored))) if te_scores else None,
        "oos_positive_folds": int(sum(1 for x in te_scores if x > 0)),
        "default_params_oos_log_return_sum": float(np.sum(base_te)) if base_te else None,
        "mean_degradation_test_over_train": (float(np.mean([f["degradation_test_over_train"] for f in scored
                                                              if f["degradation_test_over_train"] is not None]))
                                             if any(f["degradation_test_over_train"] is not None for f in scored)
                                             else None)}
    table = []
    if is_idx and oos_idx:
        for c in configs:
            res = cache[sw.key_of(c)]
            table.append({"config": c, "is_score": sw.score(res, is_idx, min_trades),
                          "oos_score": sw.score(res, oos_idx, 0),
                          "is_trades": int(sum(res[i]["trades"] for i in is_idx))})
        table.sort(key=lambda r: -r["is_score"])
        for r in table:
            r["is_score"] = r["is_score"] if math.isfinite(r["is_score"]) else None
    return {"sweep_table_by_in_sample_score": table[:15], "market": market, "segments": len(segs), "segment_rows": segs[0][1] - segs[0][0],
            "train_segments": train_segs, "test_segments": test_segs, "n_configs": len(configs),
            "n_trials_incl_neighbours": n_trials, "min_trades": min_trades,
            "folds": out_folds, "stitched_oos": stitched, "fixed_split": split,
            "default_config": base,
            "NOTE": "Selection used train segments only. The sweep is tuned against the ASSUMED market. "
                    "Winners are reported, never written to config.yaml."}

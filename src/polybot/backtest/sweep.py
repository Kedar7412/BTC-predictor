"""Parameter sweep with overfitting guards.

Design: every candidate config is simulated once on each time SEGMENT (default 91 days, bankroll
reset per segment, so segments are independent and comparable). That single pass feeds
  * selection, which may only read TRAIN segments (`select_best` takes the train indices),
  * the frozen out-of-sample evaluation of the winner on TEST segments (walkforward.py),
  * neighbourhood stability, deflation and degradation reports.
Nothing here writes to config.yaml. The winner is a report line, never an automatic setting.

Caveat that no guard removes: the sweep optimises against the ASSUMED market. It tunes the bot to
the model of the market, not to Polymarket.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

from ..config import Settings
from .jobs import SIGNAL_PARAMS, make_segments, pmap
from .scenarios import REFERENCE, market_kwargs
from .sim import SignalSet

PARAM_GRID = {
    "min_conviction": [0.58, 0.62, 0.68],
    "min_edge": [0.04, 0.06, 0.09],
    "min_ev": [0.02, 0.04, 0.07],
    "min_roi": [0.05, 0.08, 0.12],
    "market_shrink": [0.0, 0.3, 0.6],
    "kelly_fraction": [0.1, 0.25, 0.4],
    "momentum_weight": [0.0, 0.15, 0.3],
}
_N = NormalDist()
EULER = 0.5772156649


def key_of(cfg: dict) -> tuple:
    return tuple(sorted(cfg.items()))


def base_config(s: Settings) -> dict:
    return {p: getattr(s, p) for p in PARAM_GRID}


def sample_configs(s: Settings, n: int, seed: int = 0, fixed_momentum: bool = False) -> list[dict]:
    """The current settings plus n-1 random grid points (deterministic for a seed)."""
    rng = np.random.default_rng(seed)
    base = base_config(s)
    seen = {key_of(base): base}
    tries = 0
    while len(seen) < n and tries < n * 50:
        tries += 1
        cfg = {p: float(v[int(rng.integers(len(v)))]) for p, v in PARAM_GRID.items()}
        if fixed_momentum:
            cfg["momentum_weight"] = base["momentum_weight"]
        seen.setdefault(key_of(cfg), cfg)
    return list(seen.values())


def neighbours(cfg: dict, fixed_momentum: bool = False) -> list[dict]:
    """Configs that differ from cfg by one grid step in exactly one parameter."""
    out = []
    for p, vals in PARAM_GRID.items():
        if fixed_momentum and p == "momentum_weight":
            continue
        cur = cfg[p]
        i = min(range(len(vals)), key=lambda j: abs(vals[j] - cur))
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                out.append({**cfg, p: float(vals[j])})
    return out


def evaluate(get_sigs, s: Settings, configs: list[dict], segs: list[tuple[int, int]], market: dict,
             workers: int, cache: dict, label: str = "sweep", only: dict | None = None) -> None:
    """Fill cache[key] = list of per-segment dicts (None where not simulated).

    By default every segment is simulated. `only` maps key_of(config) to the segment indices that
    are actually needed (for example a neighbour is only ever scored on its winner's train
    segments), which avoids simulating segments nobody reads. Segments already cached are skipped.
    get_sigs(momentum_weight) returns the SignalSet for that model setting.
    """
    groups: dict[float, list[tuple[dict, list[int]]]] = {}
    seen: set = set()
    for c in configs:
        k = key_of(c)
        if k in seen:
            continue
        seen.add(k)
        want = sorted(only.get(k, ())) if only is not None else range(len(segs))
        have = cache.get(k)
        missing = [i for i in want if have is None or have[i] is None]
        if missing:
            groups.setdefault(float(c["momentum_weight"]), []).append((c, missing))
    for mw, items in groups.items():
        sg = get_sigs(mw)
        jobs, owners = [], []
        for c, missing in items:
            over = {p: v for p, v in c.items()}
            for si in missing:
                jobs.append({"market": market, "settings": over, "seg": segs[si]})
                owners.append((key_of(c), si))
        res = pmap(jobs, sg, s, workers, f"{label} mw={mw}")
        for (k, si), r in zip(owners, res):
            cache.setdefault(k, [None] * len(segs))[si] = {
                "trades": r["metrics"]["trades"], "log_return": r["log_return"], "pnl": r["pnl"]}


def score(seg_res: list[dict], idx: list[int], min_trades: int) -> float:
    """Mean per-segment log return over idx, or -inf if too few trades (min trade count filter)."""
    if not idx:
        return -math.inf
    if sum(seg_res[i]["trades"] for i in idx) < min_trades:
        return -math.inf
    return float(np.mean([seg_res[i]["log_return"] for i in idx]))


def select_best(cache: dict, configs: list[dict], train_idx: list[int], min_trades: int):
    """Pick the config with the best TRAIN score. Only train segments are read."""
    best, best_s = None, -math.inf
    for c in configs:
        sc = score(cache[key_of(c)], train_idx, min_trades)
        if sc > best_s:
            best, best_s = c, sc
    return best, best_s


def stability(cache: dict, cfg: dict, train_idx: list[int], min_trades: int, fixed_momentum: bool) -> dict:
    """Plateau vs spike: how do one-step neighbours score on the TRAIN segments?"""
    w = score(cache[key_of(cfg)], train_idx, min_trades)
    ns = [score(cache[key_of(n)], train_idx, min_trades) for n in neighbours(cfg, fixed_momentum)
          if key_of(n) in cache]
    fin = [x for x in ns if math.isfinite(x)]
    if not math.isfinite(w) or not fin:
        return {"winner_score": None if not math.isfinite(w) else w, "neighbours": len(ns),
                "verdict": "unknown (no scored neighbours)"}
    ratio = float(np.mean(fin) / w) if w > 0 else None
    worst = min(fin)
    if w <= 0:
        verdict = "winner is not profitable on train"
    elif ratio is not None and ratio >= 0.7 and worst > 0:
        verdict = "plateau (neighbours hold up)"
    else:
        verdict = "SPIKE (neighbours much worse or losing): likely overfit"
    return {"winner_score": w, "neighbour_mean": float(np.mean(fin)), "neighbour_worst": worst,
            "neighbour_ratio": ratio, "neighbours": len(ns), "scored_neighbours": len(fin),
            "verdict": verdict}


def expected_max_z(n_trials: int) -> float:
    """Expected maximum of n iid standard normals (Bailey and Lopez de Prado)."""
    if n_trials <= 1:
        return 0.0
    return (1 - EULER) * _N.inv_cdf(1 - 1 / n_trials) + EULER * _N.inv_cdf(1 - 1 / (n_trials * math.e))


def deflation(rets: np.ndarray, n_trials: int) -> dict:
    """Multiple-testing haircut for the winner's per-trade return series.

    z = t-statistic of mean per-trade return. After N trials the best z is expected to be about
    expected_max_z(N) even with no skill. deflated_prob = Phi(z - z0) is the probability the
    winner beats that bar. Bonferroni p = min(1, N * (1 - Phi(z))).
    """
    r = np.asarray(rets, dtype=float)
    if len(r) < 3 or r.std(ddof=1) == 0:
        return {"n_trials": n_trials, "trades": int(len(r)), "z": None, "note": "too few trades"}
    z = float(r.mean() / r.std(ddof=1) * math.sqrt(len(r)))
    z0 = expected_max_z(n_trials)
    return {"n_trials": n_trials, "trades": int(len(r)), "z": z, "expected_max_z_under_null": z0,
            "deflated_prob": float(_N.cdf(z - z0)),
            "bonferroni_p": float(min(1.0, n_trials * (1 - _N.cdf(z)))),
            "note": "deflated_prob below 0.95 means the winner is not distinguishable from the best of "
                    f"{n_trials} lucky draws"}


def winner_returns(sigs_for, s: Settings, cfg: dict, segs: list[tuple[int, int]], idx: list[int],
                   market: dict) -> np.ndarray:
    """Per-trade returns (pnl / stake) of cfg pooled over the given segments."""
    sg = sigs_for(float(cfg["momentum_weight"]))
    jobs = [{"market": market, "settings": dict(cfg), "seg": segs[i], "rets": True} for i in idx]
    res = pmap(jobs, sg, s, 1)
    parts = [r["rets"] for r in res if "rets" in r]
    return np.concatenate(parts) if parts else np.empty(0)


def segment_plan(n_rows: int, train_segs: int, test_segs: int, seg_rows: int = 91 * 288) -> list[tuple[int, int]]:
    """Segments of seg_rows windows, shrunk for short datasets so at least one fold fits."""
    seg_rows = min(seg_rows, max(288, n_rows // (train_segs + test_segs + 1)))
    return make_segments(n_rows, seg_rows)


def sweep_settings(full: bool) -> dict:
    return {"n_configs": 40 if full else 8, "min_trades": 100 if full else 10,
            "fixed_momentum": not full, "market_name": REFERENCE}


def market_for(name: str, seed: int = 0, half_spread: float = 0.01) -> dict:
    return {**market_kwargs(name), "half_spread": half_spread, "seed": seed}


__all__ = ["PARAM_GRID", "SIGNAL_PARAMS"]

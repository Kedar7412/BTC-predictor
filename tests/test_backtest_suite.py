"""Offline tests for the hard-test suite (synthetic data only)."""
import asyncio
import json

import numpy as np
import pytest

from polybot.backtest import books as B
from polybot.backtest import data as D
from polybot.backtest import jobs as J
from polybot.backtest import montecarlo as MC
from polybot.backtest import scenarios as SC
from polybot.backtest import stress as ST
from polybot.backtest import sweep as SW
from polybot.backtest import walkforward as WF
from polybot.backtest.cli import main
from polybot.backtest.market_model import AssumedMarket, NoInfoControl, TickContext
from polybot.backtest.sim import build_signals, run_sim
from polybot.config import Settings
from polybot.models import Book, MarketInfo
from polybot.recorder import Recorder


@pytest.fixture(scope="module")
def kl():
    return D.synthetic(0.12, seed=3)


@pytest.fixture(scope="module")
def sigs(kl):
    return build_signals(kl, Settings())


def test_higher_fees_never_improve_pnl(sigs):
    res = SC.run_scenarios(sigs, Settings(), workers=1, full=False, progress=False)
    for cells in res["cost_grid"].values():
        assert SC.monotonic_in_costs(cells)
    names = list(res["baseline"])
    assert names[0] == "efficient" and names[1].startswith("noisy")  # report order


def test_implausible_flag():
    ok = {"sharpe": 1.0, "final_bankroll": 1500.0, "start_bankroll": 1000.0}
    assert SC.implausible(ok) is None
    assert "artifact" in SC.implausible({**ok, "sharpe": 9.0})
    assert "artifact" in SC.implausible({**ok, "final_bankroll": 949_000.0})


def test_walkforward_windows_do_not_overlap():
    fl = WF.folds(10, 4, 1)
    assert len(fl) == 6
    for tr, te in fl:
        assert not set(tr) & set(te) and max(tr) < min(te)
    is_idx, oos = WF.split_is_oos(10)
    assert len(oos) == 3 and not set(is_idx) & set(oos) and max(is_idx) < min(oos)
    assert len(WF.split_is_oos(2)[1]) == 1


def test_selection_uses_only_train_segments():
    s = Settings()
    cfgs = SW.sample_configs(s, 6, seed=1, fixed_momentum=True)
    rng = np.random.default_rng(0)
    cache = {SW.key_of(c): [{"trades": 50, "log_return": float(rng.normal()), "pnl": 0.0} for _ in range(5)]
             for c in cfgs}
    best1, _ = SW.select_best(cache, cfgs, [0, 1, 2], 10)
    for v in cache.values():  # poison the test segments
        v[3]["log_return"], v[4]["log_return"] = 99.0, -99.0
    best2, _ = SW.select_best(cache, cfgs, [0, 1, 2], 10)
    assert best1 == best2
    # min trade filter
    for v in cache.values():
        v[0]["trades"] = v[1]["trades"] = v[2]["trades"] = 1
    assert SW.select_best(cache, cfgs, [0, 1, 2], 10)[0] is None
    assert SW.expected_max_z(100) > SW.expected_max_z(2) > 0


def test_walkforward_runs_and_reports_neighbourhood(sigs):
    s = Settings()
    wf = WF.run_walkforward(lambda mw: sigs, s, len(sigs.ws), workers=1, full=False)
    assert wf["folds"] and all("train_segments" in f for f in wf["folds"])
    f = next(f for f in wf["folds"] if f.get("config"))
    assert "verdict" in f["stability"] and "bonferroni_p" in f["deflation"] or f["deflation"].get("z") is None
    assert wf["default_config"] == SW.base_config(s)


def test_montecarlo_deterministic_and_ordered():
    rng = np.random.default_rng(1)
    pnl = rng.normal(0.5, 5.0, 400)
    day = np.arange(400) // 8
    a = MC.run_montecarlo(pnl, day, 1000.0, n_paths=200, seed=5)
    b = MC.run_montecarlo(pnl, day, 1000.0, n_paths=200, seed=5)
    assert a == b
    for mode in ("block_bootstrap", "shuffle"):
        r = a[mode]["final_return"]
        assert r["p5"] <= r["p50"] <= r["p95"]
        d = a[mode]["max_drawdown"]
        assert d["p5"] <= d["p50"] <= d["p95"]
    assert MC.run_montecarlo(pnl, day, 1000.0, n_paths=50, seed=6) != MC.run_montecarlo(pnl, day, 1000.0, n_paths=50, seed=5)
    assert MC.run_montecarlo(np.array([1.0]), np.array([0]), 1000.0)["n_paths"] == 0


def test_stress_transforms_change_data(kl):
    sub = ST.cutoff(kl, 30)
    assert len(sub) < len(kl)
    lr = lambda k: np.diff(np.log(k.close)).std()
    assert lr(ST.vol_spikes(sub, 3.0, seed=1)) > lr(sub) * 1.05
    g = ST.data_gaps(sub, 0.05, seed=1)
    assert len(g) < len(sub) * 0.97
    f = ST.flash_gaps(sub, 0.06, seed=1, every_days=5)
    assert len(f) < len(sub) and not np.allclose(f.close[:len(f)], sub.close[:len(f)])


def test_no_info_control_loses_money(sigs):
    s = Settings()
    src = NoInfoControl("noisy", noise_sd=0.5, half_spread=0.01, seed=2)
    res = run_sim(sigs, s, src)
    assert len(res.trades) > 0 and res.final_bankroll < s.bankroll


def test_signal_cache_roundtrip(kl, sigs, tmp_path):
    p = tmp_path / "s.npz"
    J.save_signals(sigs, p)
    back = J.load_signals(p)
    assert np.array_equal(back.ws, sigs.ws) and np.allclose(back.prob, sigs.prob, equal_nan=True)
    assert J.signals_key(kl, Settings()) != J.signals_key(kl, Settings(momentum_weight=0.3))
    r1 = run_sim(sigs, Settings(), AssumedMarket("noisy", 0.5)).final_bankroll
    r2 = run_sim(back, Settings(), AssumedMarket("noisy", 0.5)).final_bankroll
    assert r1 == r2


def test_signal_save_is_atomic_and_leaves_no_tmp(sigs, tmp_path, monkeypatch):
    p = tmp_path / "s.npz"
    J.save_signals(sigs, p)
    assert [f.name for f in tmp_path.iterdir()] == ["s.npz"]

    def boom(src, dst):
        raise OSError("interrupted")
    monkeypatch.setattr(J.os, "replace", boom)
    p2 = tmp_path / "t.npz"
    with pytest.raises(OSError):
        J.save_signals(sigs, p2)
    assert not p2.exists() and sorted(f.name for f in tmp_path.iterdir()) == ["s.npz"]


def test_corrupt_signal_cache_is_recomputed(kl, sigs, tmp_path):
    s = Settings()
    path = tmp_path / f"signals_{J.signals_key(kl, s)}.npz"
    path.write_bytes(b"not an npz, truncated")
    got = J.get_signals(kl, s, tmp_path, resume=True)
    assert np.array_equal(got.ws, sigs.ws)
    assert np.array_equal(J.load_signals(path).ws, sigs.ws)  # overwritten with a valid file


def test_signal_cache_key_includes_code_hash(kl, monkeypatch):
    s = Settings()
    k1 = J.signals_key(kl, s)
    monkeypatch.setattr(J, "_code_hash", lambda: "different source")
    assert J.signals_key(kl, s) != k1


def test_oos_multiple_weights_by_segment_count():
    import math
    scored = [{"test_score": 0.1, "test_segments": [3, 4]}]
    assert math.isclose(math.exp(sum(f["test_score"] * len(f["test_segments"]) for f in scored)), math.exp(0.2))


class FakeProvider:
    def __init__(self):
        self.n = 0

    async def get_market(self, ws):
        return MarketInfo(f"btc-updown-5m-{ws}", ws, ws + 300, "u", "d")

    async def get_books(self, market):
        self.n += 1
        return Book(0.40, 0.42, 100.0), Book(0.57, 0.59, 80.0)


class StepClock:
    def __init__(self, t):
        self.t = float(t)

    def now(self):
        return self.t

    async def sleep(self, s):
        self.t += s


def test_recorder_writes_and_booksource_replays(tmp_path):
    ws = 1_700_000_100
    clock = StepClock(ws + 55)
    path = tmp_path / "books.jsonl"

    async def price():
        return 65000.0
    rec = Recorder(FakeProvider(), price, path, clock, tick_seconds=5.0)
    n = asyncio.run(rec.record_window(ws))
    assert n > 40
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert {"ts", "window_start", "up_bid", "up_ask", "up_size", "down_bid", "down_ask", "down_size", "btc"} <= set(rows[0])
    src = B.RecordedBooks(path, "only")
    ctx = TickContext(0, ws, 1, ws + 120, ws + 300, 65000.0, 65010.0, float("nan"), 0.7, 0.001, 180.0)
    up, down = src.books(ctx)
    assert up.best_ask == 0.42 and down.best_bid == 0.57
    other = TickContext(1, ws + 300, 1, ws + 420, ws + 600, 65000.0, 65010.0, float("nan"), 0.7, 0.001, 180.0)
    assert src.books(other) is None  # 'only' mode skips unrecorded windows
    mixed = B.RecordedBooks(path, "mixed", AssumedMarket("efficient"))
    mixed.prepare(np.array([ws, ws + 300]))
    assert mixed.books(other) is not None and mixed.model_used == 1
    assert "REAL" in src.describe()["source"]


def test_quick_suite_smoke(tmp_path, capsys):
    out = tmp_path / "out"
    rc = main(["--suite", "quick", "--synthetic", "--years", "0.12", "--out", str(out),
               "--data-dir", str(tmp_path / "d"), "--mc-paths", "100"])
    assert rc == 0
    md = (out / "report.md").read_text()
    head = "\n".join(md.splitlines()[:12])
    assert "ASSUMPTIONS" in head and "NOT historical data" in head
    assert "1 minute" in md and "5 s" in md
    for f in ("report.html", "results.json", "trades.csv", "equity.csv", "scenarios.csv", "cost_grid.csv",
              "walkforward.csv", "sweep.csv", "montecarlo.csv", "stress.csv", "calibration.csv", "controls.csv"):
        assert (out / f).exists(), f
    R = json.loads((out / "results.json").read_text())
    assert list(R["scenarios"]["baseline"])[0] == "efficient"
    assert "PASS" in R["controls"]["verdict"] or "INCONCLUSIVE" in R["controls"]["verdict"]
    # resume reuses the cached signals
    main(["--suite", "quick", "--synthetic", "--years", "0.12", "--out", str(out),
          "--data-dir", str(tmp_path / "d"), "--mc-paths", "50", "--resume"])
    assert list((tmp_path / "d").glob("signals_*.npz"))

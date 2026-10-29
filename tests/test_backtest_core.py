import datetime as dt
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from polybot import edge as edge_mod
from polybot.backtest import data as D
from polybot.backtest import metrics as M
from polybot.backtest import sim as S
from polybot.backtest.market_model import AssumedMarket, BookSource
from polybot.config import Settings
from polybot.models import Book

T0 = 1_700_000_100  # multiple of 300


def csv_rows(n, t0=T0, unit=1000, header=False, start=30000.0):
    lines = []
    if header:
        lines.append("open_time,open,high,low,close,volume,close_time,qav,trades,tb,tq,ignore")
    for i in range(n):
        px = start + i
        lines.append(f"{(t0 + 60 * i) * unit},{px},{px + 1},{px - 1},{px + 0.5},1,0,0,0,0,0,0")
    return "\n".join(lines).encode()


def make_zip(path: Path, raw: bytes):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(path.stem + ".csv", raw)


def klines_from_closes(closes, opens=None, t0=T0):
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]]) if opens is None else np.asarray(opens, dtype=float)
    ts = t0 + 60 * np.arange(len(closes), dtype=np.int64)
    return D.clean(ts, opens, np.maximum(opens, closes), np.minimum(opens, closes), closes)


# ---------------------------------------------------------------- loader

@pytest.mark.parametrize("unit,header", [(1000, False), (1000, True), (1_000_000, False), (1_000_000, True)])
def test_parse_ms_us_header(unit, header):
    ts, o, h, l, c = D.parse_csv_bytes(csv_rows(5, unit=unit, header=header))
    assert list(ts) == [T0 + 60 * i for i in range(5)]
    assert c[0] == 30000.5 and len(o) == 5


def test_clean_quality_dup_gap():
    ts = np.array([T0, T0 + 60, T0 + 60, T0 + 300])
    px = np.array([1.0, 2.0, 3.0, 4.0])
    k = D.clean(ts, px, px, px, px)
    assert len(k) == 3 and k.quality["duplicates"] == 1
    assert k.quality["missing_minutes"] == 3 and k.quality["gap_count"] == 1
    assert k.close[1] == 3.0  # last duplicate wins


def test_cache_resume_skips_existing(tmp_path):
    calls = []

    def fetch(url, dest):
        calls.append(url)
        if url.endswith(".CHECKSUM"):
            return False
        make_zip(dest, csv_rows(100, t0=int(dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc).timestamp())))
        return True

    today = dt.date(2023, 6, 1)
    files = D.ensure_data(dt.date(2023, 1, 1), dt.date(2023, 2, 28), tmp_path, fetch, True, today)
    assert len(files) == 2 and len([c for c in calls if not c.endswith("CHECKSUM")]) == 2
    n = len(calls)
    D.ensure_data(dt.date(2023, 1, 1), dt.date(2023, 2, 28), tmp_path, fetch, True, today)
    assert len(calls) == n  # nothing re-downloaded
    assert not list(tmp_path.glob("*.zip")) and not list(tmp_path.glob("*.part"))
    k = D.load_cached(files)
    assert len(k) == 100  # both months hold the same timestamps here, deduplicated


def test_404_tolerated_and_daily_fallback(tmp_path):
    def fetch(url, dest):
        if "monthly" in url or url.endswith(".CHECKSUM"):
            return False
        if url.endswith("2023-01-02.zip"):
            return False
        make_zip(dest, csv_rows(10))
        return True

    files = D.ensure_data(dt.date(2023, 1, 1), dt.date(2023, 1, 3), tmp_path, fetch, True, dt.date(2023, 3, 1))
    assert [f.name for f in files] == ["BTCUSDT-1m-2023-01-01.npz", "BTCUSDT-1m-2023-01-03.npz"]


def test_checksum_mismatch(tmp_path):
    def fetch(url, dest):
        if url.endswith(".CHECKSUM"):
            dest.write_text("0" * 64 + "  x.zip\n")
        else:
            make_zip(dest, csv_rows(10))
        return True

    with pytest.raises(D.DataError):
        D.ensure_data(dt.date(2023, 1, 1), dt.date(2023, 1, 31), tmp_path, fetch, True, dt.date(2023, 3, 1))


# ---------------------------------------------------------------- sim helpers

def small_settings(**kw):
    base = dict(bankroll=1000.0, min_conviction=0.62, min_edge=0.06, min_ev=0.04, min_roi=0.08,
                kelly_fraction=0.25, max_stake_per_trade=50.0, max_daily_loss=100.0)
    base.update(kw)
    return Settings(**base)


class FixedBooks(BookSource):
    def books(self, ctx):
        return Book(0.49, 0.50, 1000.0), Book(0.49, 0.50, 1000.0)


def tiny_klines():
    # each window's open is the previous close (first open = first close)
    c = [100.3, 100.6, 100.9, 101.2, 101.5]      # window 1: steady rise, UP wins
    c += [101.0, 100.6, 100.3, 100.0, 99.6]      # window 2: steady fall, DOWN wins
    c += [101.0, 101.4, 100.9, 100.4, 99.0]      # window 3: jumps up, ends below its open (99.6)
    return klines_from_closes(c)


def test_window_outcome_rule():
    k = tiny_klines()
    sg = S.build_signals(k, small_settings())
    assert list(sg.up_won) == [True, False, False]  # end close > first minute open
    # equality is not an UP win
    k2 = klines_from_closes([10, 10, 10, 10, 10], [10, 10, 10, 10, 10])
    assert not S.build_signals(k2, small_settings()).up_won[0]


def test_hand_built_pnl_and_real_functions(monkeypatch):
    s = small_settings()
    sg = S.build_signals(tiny_klines(), s)
    calls = []
    real = edge_mod.best_opportunity
    monkeypatch.setattr(edge_mod, "best_opportunity", lambda *a, **kw: (calls.append(1), real(*a, **kw))[1])
    res = S.run_sim(sg, s, FixedBooks())
    assert calls  # sim goes through the live edge function
    tr = res.arrays()
    assert len(tr["pnl"]) == 3
    eff = 0.50 + s.slippage + s.fee_rate * 0.5 * 0.5
    win = 50.0 / eff - 50.0
    # w1 buys UP (wins), w2 buys DOWN (wins), w3 buys UP at first tick (loses)
    assert list(tr["side_up"]) == [True, False, True]
    np.testing.assert_allclose(tr["stake"], [50, 50, 50])
    np.testing.assert_allclose(tr["pnl"], [win, win, -50.0])
    np.testing.assert_allclose(tr["bankroll"][-1], 1000 + 2 * win - 50.0)


def test_kill_switch_is_live_risk_manager():
    s = small_settings(max_daily_loss=10.0)
    c = [100.0, 101.0, 100.0, 99.0, 98.0]  # trades UP at tick 0? falls to lose
    ws = []
    closes = []
    for _ in range(4):  # four identical losing windows on the same UTC day
        closes += [100.3, 100.6, 100.3, 100.0, 99.0]
    opens = []
    for w in range(4):
        opens += [100.0, 100.3, 100.6, 100.3, 100.0]
    sg = S.build_signals(klines_from_closes(closes, opens), s)
    res = S.run_sim(sg, s, FixedBooks())
    assert res.counters["trades"] == 1  # first loss (-50) trips the 10 USD kill switch
    assert any("kill switch" in r for r in res.counters["skip_reasons"])


def test_no_lookahead():
    rng = np.random.default_rng(1)
    closes = 30000 * np.exp(np.cumsum(rng.standard_normal(60) * 0.0005))
    s = small_settings()
    a = S.build_signals(klines_from_closes(closes), s)
    c2 = closes.copy()
    c2[34] *= 1.05      # last minute of window 6 (index 30..34)
    c2[35:] *= 0.9      # all of the future
    b = S.build_signals(klines_from_closes(c2), s)
    np.testing.assert_array_equal(a.prob[:6], b.prob[:6])
    np.testing.assert_array_equal(a.prob[6, :4], b.prob[6, :4])  # ticks only see minutes 0..3 of the window
    np.testing.assert_array_equal(a.sigma[6, :4], b.sigma[6, :4])


def test_gap_window_skipped_and_counted():
    k = klines_from_closes(np.linspace(100, 110, 20))
    keep = np.ones(len(k), dtype=bool)
    keep[7] = False
    k2 = D.clean(k.ts[keep], k.open[keep], k.high[keep], k.low[keep], k.close[keep])
    sg = S.build_signals(k2, small_settings())
    assert list(sg.valid) == [True, False, True, True]
    assert sg.stats["windows_skipped_missing_minutes"] == 1


def synth_sigs(years=0.02, seed=3, **kw):
    s = small_settings(**kw)
    return s, S.build_signals(D.synthetic(years, seed), s)


def test_prefilter_is_exact():
    s, sg = synth_sigs(min_edge=0.02, min_ev=0.01, min_roi=0.02)
    mk = lambda: AssumedMarket("stale_noisy", noise_sd=0.4, lag_minutes=2, seed=5)
    a = S.run_sim(sg, s, mk(), prefilter=True)
    b = S.run_sim(sg, s, mk(), prefilter=False)
    assert len(a.trades) > 5
    assert a.trades == b.trades and a.final_bankroll == b.final_bankroll
    assert a.counters["ticks_prefiltered"] > 0


def test_determinism_and_seed():
    s, sg = synth_sigs(min_edge=0.02, min_ev=0.01, min_roi=0.02)
    run = lambda seed: S.run_sim(sg, s, AssumedMarket("noisy", noise_sd=0.5, seed=seed))
    assert run(1).trades == run(1).trades
    assert run(1).trades != run(2).trades


def test_workers_identical():
    s = small_settings()
    k = D.synthetic(0.2, 4)
    a = S.build_signals(k, s, workers=1)
    b = S.build_signals(k, s, workers=2)
    np.testing.assert_array_equal(a.prob, b.prob)


def test_market_model_complement_and_clip():
    m = AssumedMarket("efficient", half_spread=0.01)
    m.prepare(np.array([T0]))
    from polybot.backtest.market_model import TickContext
    ctx = TickContext(0, T0, 0, T0 + 60, T0 + 300, 100.0, 100.0, float("nan"), 0.999, 1e-4, 240.0)
    up, down = m.books(ctx)
    assert up.best_ask <= 0.99 and (up.best_bid + up.best_ask) / 2 == pytest.approx(0.98)
    assert (down.best_bid + down.best_ask) / 2 == pytest.approx(0.02)
    with pytest.raises(ValueError):
        AssumedMarket("bogus")


# ---------------------------------------------------------------- metrics

def test_drawdown_known():
    eq = np.array([100, 120, 90, 100, 130.0])
    t = np.arange(5) * 86400.0
    d = M.drawdown(eq, t)
    assert d["max_drawdown"] == pytest.approx(0.25)
    assert d["max_drawdown_duration_days"] == pytest.approx(3.0)


def test_brier_and_reliability():
    p = np.array([0.9, 0.9, 0.1, 0.1])
    y = np.array([1, 0, 0, 0.0])
    assert M.brier_score(p, y) == pytest.approx((0.01 + 0.81 + 0.01 + 0.01) / 4)
    rel = M.reliability_curve(p, y, bins=10)
    assert sum(b["n"] for b in rel) == 4
    assert rel[9]["n"] == 2 and rel[9]["freq_up"] == 0.5 and rel[1]["n"] == 2 and rel[1]["freq_up"] == 0.0
    cal = M.calibration(np.array([[0.5, 0.5, 0.5, 0.5]]), np.array([True]), np.array([True]))
    assert cal["brier"] == 0.25 and cal["brier_skill_vs_0.5"] == 0.0


def test_compute_metrics_basic():
    s = small_settings()
    res = S.run_sim(S.build_signals(tiny_klines(), s), s, FixedBooks())
    m = M.compute_metrics(res.arrays(), 1000.0, T0, T0 + 900)
    assert m["trades"] == 3 and m["win_rate"] == pytest.approx(2 / 3)
    assert m["final_bankroll"] == pytest.approx(res.final_bankroll)
    assert m["profit_factor"] > 1 and set(m["pnl_by_regime"]).issubset({"low_vol", "mid_vol", "high_vol"})
    empty = M.compute_metrics(S.SimResult([], 1000.0, {}, 0.0).arrays(), 1000.0, T0, T0 + 900)
    assert empty["trades"] == 0 and empty["total_return"] == 0.0


def test_cli_synthetic_writes_files(tmp_path):
    from polybot.backtest.cli import main
    out = tmp_path / "o"
    assert main(["--synthetic", "--years", "0.02", "--out", str(out), "--config", "none"]) == 0
    for f in ("trades.csv", "summary.json", "equity.csv"):
        assert (out / f).exists()
    import json
    sm = json.loads((out / "summary.json").read_text())
    assert "MODELLED" in sm["assumed_market"]["source"] and "WARNING" in sm

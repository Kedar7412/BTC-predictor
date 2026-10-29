from polybot.config import Settings
from polybot.engine import Engine
from polybot.execution import PaperExecutor
from polybot.feeds import SimClock, SimulatedFeed
from polybot.market import SimulatedMarket
from polybot.monitor import Monitor


def build(seed=7, noise=0.15):
    s = Settings(tick_seconds=15)
    clock = SimClock(1_750_000_000 // 300 * 300 + 60)
    feed = SimulatedFeed(clock, seed=seed)
    mkt = SimulatedMarket(feed, clock, noise=noise, seed=seed)
    ex = PaperExecutor(1000)
    return s, Engine(s, feed, mkt, ex, Monitor(), clock), ex


async def test_engine_runs_and_obeys_gates():
    s, eng, ex = build()
    trades = []
    orig = eng.monitor.opened
    eng.monitor.opened = lambda p: (trades.append(p), orig(p))
    await eng.run(max_cycles=80)  # 20 minutes of virtual time, several windows
    assert eng.monitor.skips  # skipped decisions are recorded with reasons
    for p in trades:
        assert p.edge >= s.min_edge and p.prob >= s.min_conviction
    assert len({p.window_start for p in trades}) == len(trades)  # one per window
    assert ex.bankroll > 0 and not eng.open or all(p.window_end > 0 for p in eng.open)


async def test_engine_survives_errors():
    s, eng, _ = build()

    async def boom():
        raise RuntimeError("feed down")
    eng.feed.price = boom
    await eng.run(max_cycles=3)


async def test_market_lookup_retried_and_not_stale():
    s, eng, _ = build()
    real = eng.provider.get_market
    calls = {"n": 0}

    async def flaky(ws):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("gamma down")
        if calls["n"] == 2:
            return None
        return await real(ws)
    eng.provider.get_market = flaky

    async def fixed_open(ts):
        return 65000.0
    eng.feed.price_at = fixed_open
    await eng.tick()
    assert eng.market is None
    await eng.tick()
    assert eng.market is None
    await eng.tick()
    assert eng.market is not None and eng.current_window in eng.opens


async def test_no_open_price_skips_window():
    s, eng, _ = build()

    async def none(ts):
        return None
    eng.feed.price_at = none
    d = await eng.tick()
    assert d is None and eng.current_window not in eng.opens


async def test_failed_order_not_retried_in_window(monkeypatch):
    s, eng, ex = build(noise=0.0)
    attempts = {"n": 0}

    async def boom(*a):
        attempts["n"] += 1
        raise TimeoutError("post_order")
    ex.buy = boom

    async def fixed_open(ts):
        return 65000.0
    eng.feed.price_at = fixed_open
    # force a trade decision
    from polybot.models import Decision, Opportunity
    opp = Opportunity("UP", 0.9, 0.5, 0.52, 0.38, 0.3, 0.5)
    eng.risk.decide = lambda now, ws, *a, **k: (
        Decision(False, "window traded") if ws in eng.risk.traded_windows
        else Decision(True, "ok", opp, shares=10, stake=5))
    import polybot.edge as em
    monkeypatch.setattr(em, "best_opportunity", lambda *a, **k: (opp, "ok"))
    for _ in range(3):
        try:
            await eng.tick()
        except TimeoutError:
            pass
    assert attempts["n"] == 1


async def test_restore_open_positions(tmp_path):
    s, eng, ex = build()
    from polybot.models import Position
    pos = Position(eng.clock.now() // 300 * 300, eng.clock.now() // 300 * 300 + 300, "UP", "t", 10, 0.5, 5, 0.7, 0.1)
    eng.monitor.state_path = tmp_path / "st.json"
    eng.monitor.save_state(900.0, -5.0, __import__("time").strftime("%Y-%m-%d", __import__("time").gmtime(eng.clock.now())), [pos])
    s2, eng2, ex2 = build()
    eng2.monitor.state_path = eng.monitor.state_path
    eng2.restore()
    assert len(eng2.open) == 1 and eng2.open[0].shares == 10 and ex2.bankroll == 900.0

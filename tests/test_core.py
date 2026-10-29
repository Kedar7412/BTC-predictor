import math

import pytest

from polybot import edge, risk, signal
from polybot.config import LiveNotConfirmed, Settings, load_settings, require_live_confirmation
from polybot.execution import LiveExecutor, PaperExecutor
from polybot.models import Book, Decision, MarketInfo, Signal

S = Settings()
MKT = MarketInfo("m", 0, 300, "u", "d")


def sig(p):
    return Signal(p, 1e-4, 0.0, p, 1 - p)


def test_ev_known_values():
    s = Settings(fee_rate=0.0, slippage=0.0)
    o = edge.evaluate_side("UP", 0.7, 0.5, s)
    assert o.edge == pytest.approx(0.2)
    assert o.ev == pytest.approx(0.7 * 0.5 - 0.3 * 0.5)
    assert o.roi == pytest.approx(0.4)


def test_fee_reduces_ev():
    s = Settings(fee_rate=0.1, slippage=0.01)
    o = edge.evaluate_side("UP", 0.7, 0.5, s)
    assert o.eff_price == pytest.approx(0.51 + 0.1 * 0.25)
    assert o.ev < 0.7 * 0.49 - 0.3 * 0.51


def test_no_trade_below_edge():
    up, dn = Book(0.64, 0.66, 500), Book(0.33, 0.35, 500)
    opp, why = edge.best_opportunity(sig(0.68), up, dn, S)
    assert opp is None and "edge" in why


def test_no_trade_low_conviction():
    up, dn = Book(0.2, 0.22, 500), Book(0.75, 0.77, 500)
    opp, why = edge.best_opportunity(sig(0.55), up, dn, S)
    assert opp is None and "conviction" in why


def test_trade_when_all_pass_and_picks_down():
    up, dn = Book(0.7, 0.72, 500), Book(0.25, 0.27, 500)
    opp, _ = edge.best_opportunity(sig(0.2), up, dn, S)
    assert opp is not None and opp.side == "DOWN" and opp.edge >= S.min_edge and opp.ev >= S.min_ev


def test_kelly_and_caps():
    assert risk.kelly_fraction_full(0.7, 0.5) == pytest.approx(0.4)
    assert risk.kelly_fraction_full(0.4, 0.5) == 0.0
    o = edge.evaluate_side("UP", 0.9, 0.5, Settings(fee_rate=0, slippage=0))
    assert risk.size_stake(o, 1000, 1000, S) == S.max_stake_per_trade
    assert risk.size_stake(o, 1000, 10, S) == pytest.approx(10 * o.eff_price)
    small = Settings(max_stake_per_trade=1e9, kelly_fraction=0.25)
    assert risk.size_stake(o, 1000, 1e9, small) == pytest.approx(0.25 * 0.8 * 1000)


def test_signal_monotonic():
    hist = []
    a = signal.estimate(65020, 65000, 200, hist, 9e-5).prob_up
    b = signal.estimate(65040, 65000, 200, hist, 9e-5).prob_up
    assert b > a > 0.5
    late = signal.estimate(65020, 65000, 30, hist, 9e-5).prob_up
    assert late > a
    below = signal.estimate(64980, 65000, 200, hist, 9e-5).prob_up
    assert below < 0.5
    assert 0.01 <= signal.estimate(90000, 65000, 1, hist, 9e-5).prob_up <= 0.99


def opp_for(p=0.8, ask=0.5):
    return edge.evaluate_side("UP", p, ask, S)


def test_risk_kill_switch_and_windows():
    rm = risk.RiskManager(S)
    book = Book(0.49, 0.5, 500)
    d = rm.decide(150, 0, 300, book, opp_for(), 1000, 0)
    assert d.trade and d.stake > 0
    rm.mark_traded(150, 0)
    assert "already" in rm.decide(160, 0, 300, book, opp_for(), 1000, 0).reason
    assert "no-trade" in rm.decide(10, 300, 600, book, opp_for(), 1000, 0).reason
    assert "no-trade" in rm.decide(590, 300, 600, book, opp_for(), 1000, 0).reason
    assert "spread" in rm.decide(450, 300, 600, Book(0.3, 0.5, 500), opp_for(), 1000, 0).reason
    rm.record_pnl(-S.max_daily_loss)
    assert "kill" in rm.decide(450, 300, 600, book, opp_for(), 1000, 0).reason


async def test_paper_settlement():
    ex = PaperExecutor(1000)
    d = Decision(True, "ok", opp_for(), shares=100, stake=50)
    pos = await ex.buy(MKT, d, 0)
    assert ex.bankroll == pytest.approx(1000 - pos.cost)
    pnl = ex.settle(pos, up_won=True)
    assert pnl == pytest.approx(100 - pos.cost) and ex.bankroll > 1000
    pos2 = await ex.buy(MKT, d, 0)
    assert ex.settle(pos2, up_won=False) == pytest.approx(-pos2.cost)


def test_live_safety():
    assert require_live_confirmation(False, {"CONFIRM_LIVE": "yes"}) is False
    with pytest.raises(LiveNotConfirmed):
        require_live_confirmation(True, {})
    with pytest.raises(LiveNotConfirmed):
        require_live_confirmation(True, {"CONFIRM_LIVE": "no"})
    assert require_live_confirmation(True, {"CONFIRM_LIVE": "yes"}) is True
    with pytest.raises(LiveNotConfirmed):
        LiveExecutor(100, S, live_flag=False, env={"CONFIRM_LIVE": "yes", "POLYMARKET_PRIVATE_KEY": "x"})
    with pytest.raises(LiveNotConfirmed):
        LiveExecutor(100, S, live_flag=True, env={})


def test_default_config_is_paper():
    assert Settings().live is False
    assert load_settings("config.yaml").live is False


def test_cli_refuses_live(monkeypatch):
    from polybot.cli import main
    monkeypatch.delenv("CONFIRM_LIVE", raising=False)
    assert main(["--live", "--config", "config.yaml"]) == 2


def test_ev_gate_not_redundant_with_market_shrink():
    s = Settings(fee_rate=0.0, slippage=0.0, market_shrink=0.5)
    o = edge.evaluate_side("UP", 0.8, 0.5, s, mid=0.5)
    assert o.edge == pytest.approx(0.3)
    assert o.ev == pytest.approx(0.65 - 0.5)  # p_ev = 0.65
    assert o.ev < o.edge


async def test_paper_buy_none_opp():
    ex = PaperExecutor(100)
    assert await ex.buy(MKT, Decision(False, "x"), 0) is None

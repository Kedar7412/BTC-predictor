from __future__ import annotations

from typing import Protocol

from .config import LiveNotConfirmed, Settings, live_credentials, require_live_confirmation
from .models import Decision, MarketInfo, Position


class Executor(Protocol):
    bankroll: float

    async def buy(self, market: MarketInfo, decision: Decision, now: float) -> Position | None: ...

    def settle(self, pos: Position, up_won: bool) -> float: ...


class PaperExecutor:
    """Simulated fills at ask + slippage. Never touches the network."""
    live = False

    def __init__(self, bankroll: float):
        self.bankroll = bankroll
        self.pnl = 0.0

    async def buy(self, market: MarketInfo, decision: Decision, now: float) -> Position | None:
        opp = decision.opp
        if opp is None:
            return None
        cost = decision.shares * opp.eff_price
        if cost > self.bankroll or decision.shares <= 0:
            return None
        self.bankroll -= cost
        token = market.up_token if opp.side == "UP" else market.down_token
        return Position(market.start_ts, market.end_ts, opp.side, token, decision.shares,
                        opp.eff_price, cost, opp.prob, opp.edge)

    def settle(self, pos: Position, up_won: bool) -> float:
        won = (pos.side == "UP") == up_won
        payout = pos.shares if won else 0.0
        self.bankroll += payout
        pnl = payout - pos.cost
        self.pnl += pnl
        return pnl


class LiveExecutor(PaperExecutor):
    """Posts real limit orders via py-clob-client. Requires --live and CONFIRM_LIVE=yes.

    Bankroll accounting mirrors paper mode; positions settle on-chain via
    redemption, which this bot approximates using the price based outcome.
    """
    live = True

    def __init__(self, bankroll: float, settings: Settings, live_flag: bool, env: dict | None = None):
        if not require_live_confirmation(live_flag, env):
            raise LiveNotConfirmed("live executor needs --live and CONFIRM_LIVE=yes")
        super().__init__(bankroll)
        self.s = settings
        self.unreconciled: list[dict] = []
        creds = live_credentials(env)
        if not creds["key"]:
            raise LiveNotConfirmed("POLYMARKET_PRIVATE_KEY not set")
        from py_clob_client.client import ClobClient  # optional dependency
        from py_clob_client.clob_types import ApiCreds
        self.client = ClobClient("https://clob.polymarket.com", key=creds["key"], chain_id=137,
                                 signature_type=2 if creds["funder"] else 0,
                                 funder=creds["funder"] or None)
        if creds["api_key"]:
            self.client.set_api_creds(ApiCreds(creds["api_key"], creds["api_secret"], creds["api_passphrase"]))
        else:
            self.client.set_api_creds(self.client.create_or_derive_api_creds())

    async def buy(self, market: MarketInfo, decision: Decision, now: float) -> Position | None:
        import asyncio
        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY
        opp = decision.opp
        token = market.up_token if opp.side == "UP" else market.down_token
        price = round(min(opp.ask + self.s.slippage, 0.99), 2)
        size = round(decision.shares, 2)
        if size <= 0 or size * price > self.bankroll:
            return None

        def _post():
            order = self.client.create_order(OrderArgs(token_id=token, price=price, size=size, side=BUY))
            return self.client.post_order(order, OrderType.FOK)

        try:
            resp = await asyncio.to_thread(_post)
        except Exception as e:
            # Order state unknown (may have filled). Do not retry this window; flag for reconciliation.
            self.unreconciled.append({"window": market.start_ts, "token": token, "err": repr(e)})
            raise
        if not resp or not resp.get("success", False):
            return None
        cost = size * price
        self.bankroll -= cost
        return Position(market.start_ts, market.end_ts, opp.side, token, size, price, cost,
                        opp.prob, opp.edge, live=True, meta={"resp": resp})

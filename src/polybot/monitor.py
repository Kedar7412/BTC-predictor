from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("polybot")


class Monitor:
    def __init__(self, trade_log_path: str | None = None, state_path: str | None = None):
        self.trade_log = Path(trade_log_path) if trade_log_path else None
        self.state_path = Path(state_path) if state_path else None
        self.trades = 0
        self.wins = 0
        self.pnl = 0.0
        self.edges: list[float] = []
        self.skips: dict[str, int] = {}

    def _write(self, rec: dict) -> None:
        if self.trade_log:
            self.trade_log.parent.mkdir(parents=True, exist_ok=True)
            with self.trade_log.open("a") as f:
                f.write(json.dumps(rec) + "\n")

    def decision(self, ts: float, window: int, decision, prob_up: float) -> None:
        if decision.trade:
            log.info("TRADE %s p=%.3f edge=%.3f ev=%.3f stake=%.2f", decision.opp.side,
                     decision.opp.prob, decision.opp.edge, decision.opp.ev, decision.stake)
        else:
            key = decision.reason.split(" ")[0]
            self.skips[key] = self.skips.get(key, 0) + 1
            log.info("SKIP %s (p_up=%.3f)", decision.reason, prob_up)
        self._write({"type": "decision", "ts": ts, "window": window, "trade": decision.trade,
                     "reason": decision.reason, "prob_up": prob_up})

    def opened(self, pos) -> None:
        self.trades += 1
        self.edges.append(pos.edge)
        self._write({"type": "open", **{k: v for k, v in pos.__dict__.items() if k != "meta"}})

    def settled(self, pos, pnl: float, up_won: bool) -> None:
        self.pnl += pnl
        self.wins += pnl > 0
        log.info("SETTLED %s window=%d pnl=%.2f up_won=%s", pos.side, pos.window_start, pnl, up_won)
        self._write({"type": "settle", "window": pos.window_start, "side": pos.side, "pnl": pnl, "up_won": up_won})

    def stats(self) -> dict:
        return {"trades": self.trades, "win_rate": (self.wins / self.trades) if self.trades else 0.0,
                "pnl": round(self.pnl, 4),
                "avg_edge": (sum(self.edges) / len(self.edges)) if self.edges else 0.0,
                "skips": dict(self.skips)}

    def log_stats(self) -> None:
        log.info("STATS %s", json.dumps(self.stats()))

    def save_state(self, bankroll: float, daily_pnl: float, day: str, open_positions: list) -> None:
        if not self.state_path:
            return
        self.state_path.write_text(json.dumps({
            "bankroll": bankroll, "daily_pnl": daily_pnl, "day": day,
            "open_positions": [p.__dict__ for p in open_positions]}, default=str))

    def load_state(self) -> dict | None:
        if self.state_path and self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text())
            except Exception:
                return None
        return None

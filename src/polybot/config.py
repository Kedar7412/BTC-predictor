from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

import yaml


class LiveNotConfirmed(RuntimeError):
    pass


@dataclass
class Settings:
    bankroll: float = 1000.0
    tick_seconds: float = 5.0
    window_seconds: int = 300
    min_edge: float = 0.06
    min_ev: float = 0.04
    min_conviction: float = 0.62
    min_roi: float = 0.08
    market_shrink: float = 0.3  # weight of market mid in the EV probability
    fee_rate: float = 0.02
    slippage: float = 0.005
    kelly_fraction: float = 0.25
    max_stake_per_trade: float = 50.0
    max_daily_loss: float = 100.0
    max_open_positions: int = 1
    max_spread: float = 0.05
    min_liquidity_usd: float = 25.0
    no_trade_first_seconds: int = 60
    no_trade_last_seconds: int = 45
    cooldown_seconds: float = 0.0
    default_sigma_per_sqrt_s: float = 0.00009
    min_vol_samples: int = 8
    vol_window_seconds: int = 600
    momentum_weight: float = 0.15
    # runtime mode, never from yaml
    live: bool = False
    state_path: str = "state.json"
    trade_log_path: str = "logs/trades.jsonl"


def load_settings(path: str | None = None) -> Settings:
    s = Settings()
    if path and Path(path).exists():
        data = yaml.safe_load(Path(path).read_text()) or {}
        names = {f.name for f in fields(Settings)} - {"live"}
        for k, v in data.items():
            if k in names:
                setattr(s, k, type(getattr(s, k))(v))
    return s


def require_live_confirmation(live_flag: bool, env: dict | None = None) -> bool:
    """Return True only if --live was passed AND CONFIRM_LIVE=yes. Otherwise paper.

    Raises LiveNotConfirmed if --live was requested without confirmation.
    """
    env = os.environ if env is None else env
    if not live_flag:
        return False
    if env.get("CONFIRM_LIVE", "").strip().lower() != "yes":
        raise LiveNotConfirmed("--live requires CONFIRM_LIVE=yes in the environment")
    return True


def live_credentials(env: dict | None = None) -> dict:
    env = os.environ if env is None else env
    return {
        "key": env.get("POLYMARKET_PRIVATE_KEY", ""),
        "funder": env.get("POLYMARKET_FUNDER", ""),
        "api_key": env.get("POLYMARKET_API_KEY", ""),
        "api_secret": env.get("POLYMARKET_API_SECRET", ""),
        "api_passphrase": env.get("POLYMARKET_API_PASSPHRASE", ""),
    }

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "mode": "paper",
    "exchange": "kraken",
    "quote": "USD",
    "symbols": ["BTC/USD", "ETH/USD", "SOL/USD"],
    "timeframe": "4h",
    "htf_sma_days": 200,
    "strategies": ["trend", "momentum", "meanrev"],
    "heartbeat_seconds": 60,
    "db_path": "data/trader.db",
    "paper_starting_balance": 10.0,
    "target_equity": 100.0,
    "on_target": "stop",
    "risk": {
        # full = commit all available cash to the best signal(s) each time the bot enters
        # risk = size each position so a stop-out loses at most max_risk_per_trade of equity
        "allocation": "full",
        "commit_when_flat": True,
        "max_risk_per_trade": 0.02,
        "max_open_positions": 2,
        "max_position_fraction": 0.6,
        "daily_loss_limit": 0.05,
        "kill_switch_drawdown": 0.25,
        "fee_rate": 0.004,
        "slippage": 0.0005,
        "min_reward_to_fees": 3.0,
        "stop_atr_mult": 3.0,
        "target_atr_mult": 6.0,
    },
    "go_live_gate": {"min_trades": 100, "min_profit_factor": 1.3, "max_drawdown": 0.20},
    "intel": {
        "news_refresh_minutes": 15,
        "news_half_life_hours": 6,
        "max_size_boost": 0.25,
        "veto_threshold": -0.35,
        "market_veto_threshold": -0.45,
        "feeds": [
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "https://cointelegraph.com/rss",
            "https://decrypt.co/feed",
            "https://bitcoinmagazine.com/.rss/full/",
            "https://www.theblock.co/rss.xml",
            "https://www.federalreserve.gov/feeds/press_all.xml",
            "https://www.cnbc.com/id/100003114/device/rss/rss.html",
            "https://feeds.content.dowjones.io/public/rss/mw_topstories",
            "https://www.reddit.com/r/CryptoCurrency/top/.rss?t=day",
        ],
    },
    "llm": {
        "enabled": False,
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.1",
        "refresh_minutes": 30,
    },
    "dashboard": {"enabled": True, "host": "127.0.0.1", "port": 8765},
    "backtest": {"exchange": "coinbaseexchange", "days": 365},
    "learning": {
        "enabled": True,
        "max_consecutive_losses": 1,     # bench a losing symbol+strategy immediately so the next trade rotates
        "cooldown_hours": 12,
        "retune_after_losses": 2,        # losing streak (any symbol) that triggers a full walk-forward re-tune
        "retune_every_trades": 20,       # also re-tune periodically after this many closed trades
        "retune_min_hours": 12,          # never re-tune more often than this (unless the kill-switch trips)
        "tune_days": 240,                # history used for re-tuning (first 2/3 train, last 1/3 validate)
        "min_improvement": 0.01,         # new params must beat the current ones by this score margin
        "auto_resume": True,             # after the kill-switch: re-tune, wait, then resume automatically
        "auto_resume_live": False,       # live accounts need this too (a 25% drawdown is real money)
        "resume_after_hours": 24,
        "grid": {
            "stop_atr_mult": [2.0, 3.0, 4.0],
            "target_atr_mult": [4.0, 6.0, 9.0],
            "strategies": [["trend"], ["trend", "momentum"], ["trend", "momentum", "meanrev"]],
            "max_open_positions": [1, 2],
        },
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class Section(dict):
    """dict with attribute access for nested config sections."""

    def __getattr__(self, item):
        try:
            v = self[item]
        except KeyError as e:
            raise AttributeError(item) from e
        return Section(v) if isinstance(v, dict) else v


@dataclass
class Config:
    raw: dict = field(default_factory=lambda: copy.deepcopy(DEFAULTS))

    def __getattr__(self, item):
        if item == "raw":
            raise AttributeError(item)
        v = self.raw[item]
        return Section(v) if isinstance(v, dict) else v

    @property
    def base_assets(self) -> list[str]:
        return [s.split("/")[0] for s in self.symbols]


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    data: dict = {}
    if path and Path(path).exists():
        data = yaml.safe_load(Path(path).read_text()) or {}
    cfg = _merge(DEFAULTS, data)
    if overrides:
        cfg = _merge(cfg, overrides)
    if cfg["mode"] not in ("paper", "live"):
        raise ValueError("mode must be 'paper' or 'live'")
    if cfg["risk"].get("allocation", "full") not in ("full", "risk"):
        raise ValueError("risk.allocation must be 'full' or 'risk'")
    for s in cfg["symbols"]:
        if not s.endswith("/" + cfg["quote"]):
            raise ValueError(f"symbol {s} does not use quote currency {cfg['quote']}")
    return Config(cfg)

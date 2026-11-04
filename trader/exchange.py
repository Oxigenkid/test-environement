from __future__ import annotations

import os

import ccxt


def make_exchange(exchange_id: str, auth: bool = False):
    if not hasattr(ccxt, exchange_id):
        raise ValueError(f"unknown ccxt exchange id: {exchange_id}")
    params = {"enableRateLimit": True, "timeout": 20000}
    if auth:
        key, secret = os.environ.get("TRADER_API_KEY"), os.environ.get("TRADER_API_SECRET")
        if not key or not secret:
            raise RuntimeError("live mode needs TRADER_API_KEY and TRADER_API_SECRET environment variables")
        params.update(apiKey=key, secret=secret)
        if os.environ.get("TRADER_API_PASSWORD"):
            params["password"] = os.environ["TRADER_API_PASSWORD"]
    return getattr(ccxt, exchange_id)(params)


def market_limits(exchange, symbol: str) -> tuple[float, float]:
    """(min base amount, min quote cost) from the exchange's market metadata."""
    try:
        if not exchange.markets:
            exchange.load_markets()
        m = exchange.market(symbol)
        lim = m.get("limits") or {}
        return float((lim.get("amount") or {}).get("min") or 0.0), float((lim.get("cost") or {}).get("min") or 0.0)
    except Exception:
        return 0.0, 1.0

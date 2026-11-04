"""Paper and live brokers behind one interface. Positions are tracked in the journal for both."""
from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .exchange import market_limits
from .journal import Journal


@dataclass
class Fill:
    price: float
    qty: float
    fee: float


class PaperBroker:
    mode = "paper"

    def __init__(self, cfg: Config, journal: Journal, exchange=None):
        self.cfg, self.journal, self.exchange = cfg, journal, exchange
        if journal.get_state("paper_cash") is None:
            journal.set_state("paper_cash", float(cfg.paper_starting_balance))

    def limits(self, symbol: str) -> tuple[float, float]:
        return market_limits(self.exchange, symbol) if self.exchange else (0.0, 1.0)

    def precise(self, symbol: str, qty: float) -> float:
        if self.exchange is not None:
            try:
                return float(self.exchange.amount_to_precision(symbol, qty))
            except Exception:
                pass
        return float(f"{qty:.8f}")

    def cash(self) -> float:
        return float(self.journal.get_state("paper_cash"))

    def equity(self, prices: dict[str, float]) -> tuple[float, float]:
        cash = self.cash()
        held = sum(t["qty"] * prices.get(t["symbol"], t["entry_price"]) for t in self.journal.open_trades(self.mode))
        return cash + held, cash

    def buy(self, symbol: str, qty: float, price: float) -> Fill:
        r = self.cfg.risk
        px = price * (1 + r["slippage"])
        fee = px * qty * r["fee_rate"]
        cash = self.cash() - px * qty - fee
        if cash < -1e-9:
            raise RuntimeError("insufficient paper cash")
        self.journal.set_state("paper_cash", cash)
        return Fill(px, qty, fee)

    def sell(self, symbol: str, qty: float, price: float) -> Fill:
        r = self.cfg.risk
        px = price * (1 - r["slippage"])
        fee = px * qty * r["fee_rate"]
        self.journal.set_state("paper_cash", self.cash() + px * qty - fee)
        return Fill(px, qty, fee)


class LiveBroker:
    """Real orders through ccxt. Use API keys with trade permission only (never withdrawal)."""

    mode = "live"

    def __init__(self, cfg: Config, journal: Journal, exchange):
        self.cfg, self.journal, self.ex = cfg, journal, exchange
        self.ex.load_markets()

    def limits(self, symbol: str) -> tuple[float, float]:
        return market_limits(self.ex, symbol)

    def precise(self, symbol: str, qty: float) -> float:
        return float(self.ex.amount_to_precision(symbol, qty))

    def cash(self) -> float:
        return float(self.ex.fetch_balance()["free"].get(self.cfg.quote, 0.0) or 0.0)

    def equity(self, prices: dict[str, float]) -> tuple[float, float]:
        bal = self.ex.fetch_balance()
        total = bal.get("total", {})
        cash = float(bal["free"].get(self.cfg.quote, 0.0) or 0.0)
        eq = float(total.get(self.cfg.quote, 0.0) or 0.0)
        for sym in self.cfg.symbols:
            base = sym.split("/")[0]
            eq += float(total.get(base, 0.0) or 0.0) * prices.get(sym, 0.0)
        return eq, cash

    def _fill(self, order: dict, symbol: str, side: str, qty: float, price: float) -> Fill:
        try:
            order = self.ex.fetch_order(order["id"], symbol) or order
        except Exception:
            pass
        px = float(order.get("average") or order.get("price") or price)
        filled = float(order.get("filled") or qty)
        fee_info = order.get("fee") or {}
        if fee_info.get("cost") is not None and fee_info.get("currency") == self.cfg.quote:
            fee = float(fee_info["cost"])
        else:
            fee = px * filled * self.cfg.risk["fee_rate"]
        return Fill(px, filled, fee)

    def buy(self, symbol: str, qty: float, price: float) -> Fill:
        amount = self.precise(symbol, qty)
        order = self.ex.create_order(symbol, "market", "buy", amount)
        return self._fill(order, symbol, "buy", amount, price)

    def sell(self, symbol: str, qty: float, price: float) -> Fill:
        base = symbol.split("/")[0]
        free = float(self.ex.fetch_balance()["free"].get(base, 0.0) or 0.0)
        amount = self.precise(symbol, min(qty, free))
        if amount <= 0:
            raise RuntimeError(f"no {base} balance to sell")
        order = self.ex.create_order(symbol, "market", "sell", amount)
        return self._fill(order, symbol, "sell", amount, price)

    def convert_holdings_to_quote(self, prices: dict[str, float]) -> list[str]:
        """Sell deposited coins of the tracked symbols into the quote currency so the bot starts flat."""
        done = []
        bal = self.ex.fetch_balance()["free"]
        for sym in self.cfg.symbols:
            base = sym.split("/")[0]
            amt = float(bal.get(base, 0.0) or 0.0)
            min_amt, min_cost = self.limits(sym)
            if amt > 0 and amt >= min_amt and amt * prices.get(sym, 0) >= min_cost:
                fill = self.sell(sym, amt, prices.get(sym, 0))
                done.append(f"sold {fill.qty} {base} at {fill.price:.2f}")
        return done

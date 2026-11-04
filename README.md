# Autonomous trader

A local, multi-agent crypto trading bot that follows the capital-ladder plan: start with $10 or more,
trade under hard risk limits, and stop when the account reaches the start of ladder stage 02 ($100).

It runs in **paper mode by default**. Live trading stays locked until the paper record passes the go-live gate.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp config.example.yaml config.yaml          # optional, every value has a default
.venv/bin/python -m trader run              # paper trading + dashboard on http://127.0.0.1:8765
```

Other commands:

| Command | What it does |
|---|---|
| `python -m trader status` | account state, regimes, intel, learned params and the go-live gate |
| `python -m trader improve` | walk-forward re-tune on recent history (`--dry-run` to preview without adopting) |
| `python -m trader lessons` | print post-mortem diagnoses, active cool-downs, and re-tune history |
| `python -m trader reset-learning` | clear learned parameters and cool-downs back to `config.yaml` |
| `python -m trader backtest --days 730` | replay the strategies on historical candles (`--compare 1h,4h`) |
| `python -m trader news` | scrape the feeds now and print scored headlines |
| `python -m trader dashboard` | serve the read-only dashboard only |
| `python -m trader resume` | clear a kill-switch halt |
| `python -m trader reset-goal` / `reset-paper` | clear the target flag / wipe the paper account |
| `python -m trader convert-deposit` | live only: sell deposited coins into the quote currency |
| `python -m trader run --live` | live trading (refused unless the gate passes) |

## How it works

A supervisor wakes every `heartbeat_seconds` (default 60) and runs the agents in order. A failing agent is
logged and skipped; it never crashes the loop. All state lives in SQLite (`data/trader.db`), so restarts resume.

1. **Market data**: live prices every beat; closed 4h candles + indicators when a bar completes; daily candles once a day.
2. **News**: RSS from CoinDesk, Cointelegraph, Decrypt, Bitcoin Magazine, The Block, the Fed, CNBC, MarketWatch;
   VADER + crypto lexicon sentiment, tagged per coin, decayed with a 6h half-life.
3. **Mood**: Crypto Fear & Greed index and CoinGecko global market cap change.
4. **LLM (optional)**: a local OpenAI-compatible model (e.g. Ollama) scores the headlines.
5. **Intel fusion**: turns the above into a bounded size multiplier (max +25%) or a veto on new buys.
6. **Risk guardian**: equity, peak, daily loss limit, kill-switch, target, and automatic resume after cool-off.
7. **Strategist**: regime detection and the strategy ensemble, weighted by each strategy's recent results and filtered by learner cool-downs.
8. **Execution**: commits available capital (`risk.allocation: full` by default), manages stops, targets, trailing stops, and exits.
9. **Learner**: runs post-mortems on losing trades, benches losing symbol+strategy pairs on cool-down, and walk-forward re-tunes parameters when a losing streak or kill-switch occurs.

### Regimes and strategies

- Long-term filter: new buys only when the last completed **daily close is above its 200-day average**
  (`htf_sma_days`). Below it the regime is `bear` and the bot stays in cash.
- `trend_up`: trend (Donchian breakout) and momentum (pullback reclaim). `high_vol`: trend only.
  `range`: mean reversion (RSI + Bollinger). `trend_down`, `crisis`, `bear`: no new buys.
- Strategies can be switched off with `strategies: [trend]` etc.

News can shrink, block, or slightly enlarge a trade. It can never exceed the risk limits.

## Allocation and Risk limits (config `risk:` and `learning:`)

| Setting | Default |
|---|---|
| Allocation mode (`risk.allocation`) | `full` (commits all available cash across the best entry signal(s) on the beat; set `risk` for stop-based 2%-risk sizing) |
| Max open positions | 2 (can be adapted to 1 or 2 by the learner) |
| Daily loss limit (no new entries until next UTC day) | 5% |
| Kill-switch (flatten everything and halt) | 25% below peak equity |
| Expected gain must be at least | 3x round-trip fees |
| Stop / target | 3 ATR / 6 ATR, trailing stop 3 ATR (adapted by the learner within configured grid) |
| Self-improvement (`learning:`) | Post-mortem on every loss, 48h cool-down after 3 consecutive losses on a pair, walk-forward re-tune after 2 losses or kill-switch, auto-resume after 24h in paper mode |

## Going live

1. Run paper mode until `status` shows the gate as passed: 100+ trades, profit factor >= 1.3, drawdown < 20%.
2. Create exchange API keys with **trade permission only, no withdrawals**.
3. Export them, never put them in a file in this folder:
   `export TRADER_API_KEY=... TRADER_API_SECRET=...` (`TRADER_API_PASSWORD` if the exchange needs one).
4. Deposit, run `python -m trader convert-deposit`, then `python -m trader run --live`.

`--override-gate` skips the gate. Don't use it with money you can't lose.

Small accounts: exchange minimum order sizes apply. On Kraken, a SOL order needs 0.06 SOL, which can exceed the
60% position cap of a $10 account; the bot then skips that trade instead of breaking its limits.

## Backtest results (Coinbase data to Oct 2026, $10 start, fees 0.4%/side, no news)

| Setup | 365 days | 730 days |
|---|---|---|
| 1h, no daily filter (first version) | -24% (kill-switch) | -21% (kill-switch) |
| 4h + 200-day filter, all strategies (default) | -9.3%, max DD 11% | -1.2%, max DD 14% |
| 4h + 200-day filter, trend only | -3.9%, max DD 8% | +6.2%, max DD 9%, PF 1.31 |
| Buy and hold BTC / ETH / SOL | -27% / -35% / -41% | +36% / +4% / -23% |

No configuration shows an edge large enough to turn $10 into $100 quickly. The bot mostly protects capital in
falling markets. Past results don't predict future results. This is not financial advice.

## Tests

```bash
.venv/bin/python -m pytest -q
```

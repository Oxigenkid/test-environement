"""SQLite journal shared by every agent and the dashboard."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mode TEXT NOT NULL,
    symbol TEXT NOT NULL,
    strategy TEXT NOT NULL,
    regime TEXT,
    entry_ts REAL NOT NULL,
    entry_price REAL NOT NULL,
    qty REAL NOT NULL,
    stop REAL NOT NULL,
    target REAL NOT NULL,
    trail_atr REAL,
    highest REAL,
    entry_fee REAL DEFAULT 0,
    exit_ts REAL,
    exit_price REAL,
    exit_fee REAL DEFAULT 0,
    pnl REAL,
    exit_reason TEXT,
    intel TEXT
);
CREATE TABLE IF NOT EXISTS equity (ts REAL NOT NULL, mode TEXT NOT NULL, equity REAL NOT NULL, cash REAL NOT NULL);
CREATE TABLE IF NOT EXISTS news (
    link TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source TEXT,
    published REAL NOT NULL,
    fetched REAL NOT NULL,
    score REAL NOT NULL,
    symbols TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (ts REAL NOT NULL, level TEXT NOT NULL, agent TEXT NOT NULL, message TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS lessons (
    ts REAL NOT NULL, mode TEXT NOT NULL, trade_id INTEGER, symbol TEXT, strategy TEXT, regime TEXT,
    kind TEXT NOT NULL, lesson TEXT NOT NULL, pnl REAL, loss_pct REAL, mfe REAL
);
CREATE TABLE IF NOT EXISTS tunings (
    ts REAL NOT NULL, reason TEXT NOT NULL, adopted INTEGER NOT NULL, current_score REAL, best_score REAL,
    old_params TEXT, new_params TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_open ON trades(mode, exit_ts);
CREATE INDEX IF NOT EXISTS idx_news_pub ON news(published);
"""


class Journal:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    # state -------------------------------------------------------------
    def get_state(self, key: str, default: Any = None) -> Any:
        rows = self.query("SELECT value FROM state WHERE key=?", (key,))
        return json.loads(rows[0]["value"]) if rows else default

    def set_state(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT INTO state(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )

    # events ------------------------------------------------------------
    def log(self, agent: str, message: str, level: str = "info") -> None:
        self.execute("INSERT INTO events VALUES(?,?,?,?)", (time.time(), level, agent, message))
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {level.upper():5} {agent:10} {message}", flush=True)

    # trades ------------------------------------------------------------
    def open_trades(self, mode: str) -> list[dict]:
        return self.query("SELECT * FROM trades WHERE mode=? AND exit_ts IS NULL ORDER BY id", (mode,))

    def closed_trades(self, mode: str, limit: int | None = None) -> list[dict]:
        sql = "SELECT * FROM trades WHERE mode=? AND exit_ts IS NOT NULL ORDER BY exit_ts DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return self.query(sql, (mode,))

    def record_equity(self, mode: str, equity: float, cash: float) -> None:
        self.execute("INSERT INTO equity VALUES(?,?,?,?)", (time.time(), mode, equity, cash))

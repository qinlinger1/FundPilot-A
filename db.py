from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


def default_db_path() -> str:
    base = Path(__file__).resolve().parent
    data_dir = base / "data"
    os.makedirs(data_dir, exist_ok=True)
    return str(data_dir / "dashboard.db")


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS holdings (
            fund_name TEXT PRIMARY KEY,
            fund_code TEXT,
            track TEXT,
            etf_code TEXT,
            position_value REAL,
            cost REAL,
            pnl REAL,
            holding_days INTEGER,
            allow_add INTEGER,
            last_add_date TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_date TEXT,
            decision_time TEXT,
            action TEXT,
            target_fund TEXT,
            amount_or_ratio TEXT,
            buy_score REAL,
            sell_score REAL,
            reason_json TEXT,
            forbidden_json TEXT,
            meta_json TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_date TEXT,
            target_fund TEXT,
            etf_pct REAL,
            nav_pct REAL,
            consistency TEXT,
            note TEXT
        )
        """
    )
    conn.commit()


def load_holdings(conn: sqlite3.Connection) -> pd.DataFrame:
    try:
        return pd.read_sql_query("SELECT * FROM holdings", conn)
    except Exception:
        return pd.DataFrame(
            columns=[
                "fund_name",
                "fund_code",
                "track",
                "etf_code",
                "position_value",
                "cost",
                "pnl",
                "holding_days",
                "allow_add",
                "last_add_date",
            ]
        )


def upsert_holdings(conn: sqlite3.Connection, holdings_df: pd.DataFrame) -> None:
    cols = [
        "fund_name",
        "fund_code",
        "track",
        "etf_code",
        "position_value",
        "cost",
        "pnl",
        "holding_days",
        "allow_add",
        "last_add_date",
    ]
    df = holdings_df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]

    with conn:
        conn.execute("DELETE FROM holdings")
        conn.executemany(
            """
            INSERT OR REPLACE INTO holdings
            (fund_name, fund_code, track, etf_code, position_value, cost, pnl, holding_days, allow_add, last_add_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(r["fund_name"]) if r["fund_name"] is not None else "",
                    r["fund_code"],
                    r["track"],
                    r["etf_code"],
                    float(r["position_value"]) if r["position_value"] not in (None, "") else 0.0,
                    float(r["cost"]) if r["cost"] not in (None, "") else None,
                    float(r["pnl"]) if r["pnl"] not in (None, "") else None,
                    int(r["holding_days"]) if r["holding_days"] not in (None, "") else None,
                    int(r["allow_add"]) if r["allow_add"] not in (None, "") else 1,
                    r["last_add_date"],
                )
                for _, r in df.iterrows()
                if str(r["fund_name"]).strip()
            ],
        )


def _json_dumps(v) -> str:
    if v is None:
        return "null"
    if is_dataclass(v):
        v = asdict(v)
    return json.dumps(v, ensure_ascii=False, default=str)


def save_decision(
    conn: sqlite3.Connection,
    *,
    dt: datetime,
    action: str,
    target_fund: str | None,
    amount_or_ratio: str | None,
    buy_score: float | None,
    sell_score: float | None,
    reason: dict,
    forbidden: list[str],
    meta: dict | None = None,
) -> None:
    with conn:
        conn.execute(
            """
            INSERT INTO decisions
            (decision_date, decision_time, action, target_fund, amount_or_ratio, buy_score, sell_score,
             reason_json, forbidden_json, meta_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dt.date().isoformat(),
                dt.strftime("%H:%M:%S"),
                action,
                target_fund,
                amount_or_ratio,
                buy_score,
                sell_score,
                _json_dumps(reason),
                _json_dumps(forbidden),
                _json_dumps(meta or {}),
            ),
        )


def load_decisions(conn: sqlite3.Connection, limit: int = 200) -> pd.DataFrame:
    try:
        return pd.read_sql_query(
            "SELECT * FROM decisions ORDER BY id DESC LIMIT ?",
            conn,
            params=(limit,),
        )
    except Exception:
        return pd.DataFrame()


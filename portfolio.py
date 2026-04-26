from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from config import (
    DEFAULT_CASH_BUFFER_RATIO,
    DEFAULT_PRINCIPAL_CNY,
    MAX_CONSECUTIVE_ADD_DAYS,
    MAX_SINGLE_FUND_RATIO,
    MAX_SINGLE_TRACK_RATIO,
)


HOLDING_COLUMNS = [
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


def normalize_holdings(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=HOLDING_COLUMNS)
    out = df.copy()
    for c in HOLDING_COLUMNS:
        if c not in out.columns:
            out[c] = None
    out["position_value"] = pd.to_numeric(out["position_value"], errors="coerce").fillna(0.0)
    out["allow_add"] = pd.to_numeric(out["allow_add"], errors="coerce").fillna(1).astype(int)
    return out[HOLDING_COLUMNS]


@dataclass(frozen=True)
class PortfolioState:
    principal: float
    cash: float
    holdings_value: float
    holdings: pd.DataFrame


def compute_portfolio_state(principal: float, holdings_df: pd.DataFrame) -> PortfolioState:
    principal2 = float(principal) if principal not in (None, "") else DEFAULT_PRINCIPAL_CNY
    h = normalize_holdings(holdings_df)
    holdings_value = float(h["position_value"].sum()) if not h.empty else 0.0
    cash = max(0.0, principal2 - holdings_value)
    return PortfolioState(principal=principal2, cash=cash, holdings_value=holdings_value, holdings=h)


def fund_position_ratio(state: PortfolioState, fund_name: str) -> float:
    df = state.holdings
    if df.empty:
        return 0.0
    row = df[df["fund_name"] == fund_name]
    if row.empty:
        return 0.0
    v = float(row.iloc[0]["position_value"] or 0.0)
    if state.principal <= 0:
        return 0.0
    return v / state.principal


def track_position_ratio(state: PortfolioState, track: str) -> float:
    df = state.holdings
    if df.empty:
        return 0.0
    v = float(df[df["track"] == track]["position_value"].sum())
    if state.principal <= 0:
        return 0.0
    return v / state.principal


def cash_buffer_ok(state: PortfolioState, buy_amount: float) -> bool:
    min_cash = state.principal * DEFAULT_CASH_BUFFER_RATIO
    return (state.cash - buy_amount) >= min_cash


def can_add_more(
    state: PortfolioState,
    *,
    fund_name: str,
    track: str,
    buy_amount: float,
    today: date,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    if state.principal <= 0:
        reasons.append("本金为0，无法计算仓位")
        return False, reasons

    new_fund_ratio = (fund_position_ratio(state, fund_name) * state.principal + buy_amount) / state.principal
    if new_fund_ratio > MAX_SINGLE_FUND_RATIO:
        reasons.append("单只基金仓位将超过25%上限")

    new_track_ratio = (track_position_ratio(state, track) * state.principal + buy_amount) / state.principal
    if new_track_ratio > MAX_SINGLE_TRACK_RATIO:
        reasons.append("单一赛道仓位将超过30%上限")

    if buy_amount > state.cash:
        reasons.append("现金不足")

    if not cash_buffer_ok(state, buy_amount):
        reasons.append("触发现金安全垫限制（至少保留20%现金）")

    df = state.holdings
    if not df.empty:
        r = df[df["fund_name"] == fund_name]
        if not r.empty:
            allow_add = int(r.iloc[0]["allow_add"] or 0)
            if allow_add == 0:
                reasons.append("该基金标记为不允许加仓")
            last_add_date = r.iloc[0].get("last_add_date")
            if last_add_date:
                try:
                    last_d = pd.to_datetime(last_add_date).date()
                    if (today - last_d).days <= MAX_CONSECUTIVE_ADD_DAYS:
                        reasons.append("连续加仓限制：同一只基金不能连续3个交易日加仓")
                except Exception:
                    pass

    return len(reasons) == 0, reasons


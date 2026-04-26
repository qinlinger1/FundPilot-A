from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas as pd


CN_TZ = ZoneInfo("Asia/Shanghai")


def now_cn() -> datetime:
    return datetime.now(tz=CN_TZ)


@lru_cache(maxsize=1)
def get_trade_calendar() -> set[date]:
    try:
        import akshare as ak  # type: ignore
    except Exception:
        return set()

    try:
        df = ak.tool_trade_date_hist_sina()
        col = next((c for c in df.columns if "trade" in c.lower() and "date" in c.lower()), None)
        if col is None:
            col = df.columns[0]
        s = pd.to_datetime(df[col]).dt.date
        return set(s.tolist())
    except Exception:
        return set()


def is_trade_day(d: date) -> bool:
    cal = get_trade_calendar()
    if not cal:
        return d.weekday() < 5
    return d in cal


@dataclass(frozen=True)
class TradingStatus:
    now: datetime
    is_trade_day: bool
    phase: str
    is_intraday: bool
    is_final_window: bool
    is_execute_only: bool
    is_after_close: bool
    is_before_open: bool


def get_trading_status(dt: datetime) -> TradingStatus:
    d = dt.date()
    td = is_trade_day(d)
    t = dt.timetz().replace(tzinfo=None)

    before_open = t < time(9, 30)
    in_morning = time(9, 30) <= t < time(11, 30)
    noon = time(11, 30) <= t < time(13, 0)
    in_afternoon = time(13, 0) <= t < time(14, 50)
    final_window = time(14, 50) <= t < time(14, 57)
    execute_only = time(14, 57) <= t < time(15, 0)
    after_close = t >= time(15, 0)

    if before_open:
        phase = "开盘前"
    elif in_morning:
        phase = "早盘观察"
    elif noon:
        phase = "午间观察"
    elif in_afternoon:
        phase = "盘中观察"
    elif final_window:
        phase = "终判区间"
    elif execute_only:
        phase = "仅执行区间"
    else:
        phase = "盘后复盘"

    intraday = in_morning or in_afternoon or final_window or execute_only

    return TradingStatus(
        now=dt,
        is_trade_day=td,
        phase=phase,
        is_intraday=intraday and td,
        is_final_window=final_window and td,
        is_execute_only=execute_only and td,
        is_after_close=after_close,
        is_before_open=before_open,
    )


def format_currency(v: float | int | None) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):,.0f}元"
    except Exception:
        return "-"


def format_pct(v: float | int | None) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v) * 100:.2f}%"
    except Exception:
        return "-"


def safe_float(v) -> float | None:
    try:
        if v is None:
            return None
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def safe_div(a: float | None, b: float | None) -> float | None:
    a2 = safe_float(a)
    b2 = safe_float(b)
    if a2 is None or b2 is None or b2 == 0:
        return None
    return a2 / b2


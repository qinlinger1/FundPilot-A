from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FetchResult:
    ok: bool
    df: pd.DataFrame | None
    error: str | None


def _standardize_ohlc_df(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for c in df.columns:
        c2 = str(c).strip()
        if c2 in ("日期", "时间", "交易日期", "净值日期"):
            rename_map[c] = "date"
        elif c2 in ("开盘", "开盘价", "open"):
            rename_map[c] = "open"
        elif c2 in ("收盘", "收盘价", "close"):
            rename_map[c] = "close"
        elif c2 in ("最高", "最高价", "high"):
            rename_map[c] = "high"
        elif c2 in ("最低", "最低价", "low"):
            rename_map[c] = "low"
        elif c2 in ("成交量", "volume"):
            rename_map[c] = "volume"
        elif c2 in ("成交额", "amount"):
            rename_map[c] = "amount"
        elif c2 in ("涨跌幅", "日增长率", "pct_chg", "涨跌幅(%)"):
            rename_map[c] = "pct_chg"
    out = df.rename(columns=rename_map).copy()

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for c in ("open", "close", "high", "low", "volume", "amount", "pct_chg"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        else:
            out[c] = np.nan
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return out[["date", "open", "close", "high", "low", "volume", "amount", "pct_chg"]]


def _safe_call(fn: Callable, **kwargs) -> FetchResult:
    try:
        df = fn(**kwargs)
        if df is None or (hasattr(df, "empty") and df.empty):
            return FetchResult(ok=False, df=None, error="返回数据为空")
        return FetchResult(ok=True, df=df, error=None)
    except Exception as e:
        return FetchResult(ok=False, df=None, error=str(e))


def _guess_cn_symbol_prefix(code: str) -> str | None:
    c = str(code).strip().lower()
    if not c:
        return None
    if c.startswith(("sh", "sz")):
        return c[:2]
    if len(c) == 6 and c.isdigit():
        if c.startswith(("0", "3", "1")):
            return "sz"
        return "sh"
    return None


def _etf_sina_symbols(etf_code: str) -> list[str]:
    c = str(etf_code).strip()
    if not c:
        return []
    c_low = c.lower()
    if c_low.startswith(("sh", "sz")):
        return [c_low]
    p = _guess_cn_symbol_prefix(c)
    if p is None:
        return [c]
    return [f"{p}{c}"]


def fetch_etf_daily(etf_code: str, start: date | None = None) -> FetchResult:
    try:
        import akshare as ak  # type: ignore
    except Exception as e:
        return FetchResult(ok=False, df=None, error=f"AkShare 导入失败: {e}")

    start_date = None
    if start is not None:
        start_date = start.strftime("%Y%m%d")

    candidates: list[tuple[Callable, dict]] = []

    fn_em = getattr(ak, "fund_etf_hist_em", None)
    if callable(fn_em):
        candidates.append(
            (
                fn_em,
                {"symbol": etf_code, "period": "daily", "start_date": start_date, "end_date": None, "adjust": ""},
            )
        )

    fn_sina = getattr(ak, "fund_etf_hist_sina", None)
    if callable(fn_sina):
        for sym in _etf_sina_symbols(etf_code):
            candidates.append((fn_sina, {"symbol": sym}))

    last_err = None
    for fn, kw in candidates:
        kw = {k: v for k, v in kw.items() if v is not None}
        r = _safe_call(fn, **kw)
        if r.ok and r.df is not None:
            try:
                return FetchResult(ok=True, df=_standardize_ohlc_df(r.df), error=None)
            except Exception as e:
                last_err = f"字段标准化失败: {e}"
        else:
            last_err = r.error
    return FetchResult(ok=False, df=None, error=last_err or "数据获取失败")


def fetch_etf_spot(etf_code: str) -> FetchResult:
    try:
        import akshare as ak  # type: ignore
    except Exception as e:
        return FetchResult(ok=False, df=None, error=f"AkShare 导入失败: {e}")

    candidates: list[tuple[Callable, dict]] = []
    fn_em = getattr(ak, "fund_etf_spot_em", None)
    if callable(fn_em):
        candidates.append((fn_em, {}))
    fn_sina = getattr(ak, "fund_etf_spot_sina", None)
    if callable(fn_sina):
        candidates.append((fn_sina, {}))

    last_err = None
    for fn, kw in candidates:
        r = _safe_call(fn, **kw)
        if not r.ok or r.df is None:
            last_err = r.error
            continue
        df = r.df.copy()
        code_col = next((c for c in df.columns if str(c) in ("代码", "symbol", "基金代码")), None)
        if code_col is None:
            last_err = "无法识别实时行情代码列"
            continue
        target = str(etf_code).strip().lower()
        targets = {target}
        for sym in _etf_sina_symbols(target):
            targets.add(sym.lower())
        s = df[code_col].astype(str).str.strip().str.lower()
        row = df[s.isin(targets) | s.str.endswith(target)]
        if row.empty:
            last_err = "实时行情未找到该 ETF"
            continue
        return FetchResult(ok=True, df=row.reset_index(drop=True), error=None)
    return FetchResult(ok=False, df=None, error=last_err or "实时行情获取失败")


def fetch_index_daily(index_symbol: str, start: date | None = None) -> FetchResult:
    try:
        import akshare as ak  # type: ignore
    except Exception as e:
        return FetchResult(ok=False, df=None, error=f"AkShare 导入失败: {e}")

    start_str = start.strftime("%Y%m%d") if start is not None else None
    candidates: list[tuple[Callable, dict]] = []
    fn1 = getattr(ak, "stock_zh_index_daily", None)
    if callable(fn1):
        candidates.append((fn1, {"symbol": index_symbol}))
    fn2 = getattr(ak, "stock_zh_index_daily_em", None)
    if callable(fn2):
        candidates.append((fn2, {"symbol": index_symbol}))
    fn3 = getattr(ak, "index_zh_a_hist", None)
    if callable(fn3):
        candidates.append((fn3, {"symbol": index_symbol, "period": "daily", "start_date": start_str, "end_date": None}))

    last_err = None
    for fn, kw in candidates:
        kw = {k: v for k, v in kw.items() if v is not None}
        r = _safe_call(fn, **kw)
        if r.ok and r.df is not None:
            try:
                return FetchResult(ok=True, df=_standardize_ohlc_df(r.df), error=None)
            except Exception as e:
                last_err = f"字段标准化失败: {e}"
        else:
            last_err = r.error
    return FetchResult(ok=False, df=None, error=last_err or "指数数据获取失败")


def fetch_open_fund_nav(fund_code: str) -> FetchResult:
    try:
        import akshare as ak  # type: ignore
    except Exception as e:
        return FetchResult(ok=False, df=None, error=f"AkShare 导入失败: {e}")

    if not fund_code:
        return FetchResult(ok=False, df=None, error="基金代码为空")

    candidates: list[tuple[Callable, dict]] = [
        (ak.fund_open_fund_info_em, {"fund": fund_code, "indicator": "单位净值走势"}),
        (ak.fund_open_fund_info_em, {"fund": fund_code, "indicator": "累计净值走势"}),
    ]

    last_err = None
    for fn, kw in candidates:
        r = _safe_call(fn, **kw)
        if not r.ok or r.df is None:
            last_err = r.error
            continue
        df = r.df.copy()
        rename_map = {}
        for c in df.columns:
            c2 = str(c).strip()
            if c2 in ("净值日期", "日期"):
                rename_map[c] = "date"
            elif c2 in ("单位净值", "累计净值", "净值"):
                rename_map[c] = "nav"
            elif c2 in ("日增长率", "涨跌幅"):
                rename_map[c] = "pct_chg"
        out = df.rename(columns=rename_map).copy()
        if "date" not in out.columns or "nav" not in out.columns:
            last_err = "净值字段解析失败"
            continue
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["nav"] = pd.to_numeric(out["nav"], errors="coerce")
        if "pct_chg" in out.columns:
            out["pct_chg"] = pd.to_numeric(out["pct_chg"], errors="coerce") / 100.0
        else:
            out["pct_chg"] = np.nan
        out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return FetchResult(ok=True, df=out[["date", "nav", "pct_chg"]], error=None)

    return FetchResult(ok=False, df=None, error=last_err or "基金净值获取失败")

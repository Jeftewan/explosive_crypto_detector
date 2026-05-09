"""
Parallel downloads from Binance Futures API.
All functions return a pandas DataFrame with a UTC datetime index or column.
"""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd

from .binance_client import BinanceClient
from ..config import (
    BINANCE_BASE_URL,
    KLINES_DAYS,
    FR_DAYS,
    OI_DAYS,
    LS_DAYS,
    TAKER_DAYS,
    DEFAULT_KLINE_INTERVAL,
    MIN_VOLUME_USDT,
    DEFAULT_TOP_SYMBOLS,
    MIN_KLINE_DAYS,
)

logger = logging.getLogger(__name__)

# Binance klines max 1500 rows per request
_KLINE_LIMIT = 1500
# Funding rate max 1000 rows per request
_FR_LIMIT = 1000
# OI / L/S / Taker max 500 rows per request
_OI_LIMIT = 500


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _now_ms() -> int:
    return _ms(datetime.now(timezone.utc))


def _days_ago_ms(days: int) -> int:
    return _ms(datetime.now(timezone.utc) - timedelta(days=days))


# ─── Symbol universe ──────────────────────────────────────────────────────────

def get_perp_symbols(
    client: BinanceClient,
    min_volume: float = MIN_VOLUME_USDT,
    top: int = DEFAULT_TOP_SYMBOLS,
) -> list[str]:
    """Return top USDT-perpetual symbols sorted by 24h quote volume."""
    data = client.get("/fapi/v1/ticker/24hr")
    symbols = []
    for item in data:
        sym = item.get("symbol", "")
        vol = float(item.get("quoteVolume", 0))
        if sym.endswith("USDT") and vol >= min_volume:
            symbols.append((sym, vol))
    symbols.sort(key=lambda x: x[1], reverse=True)
    result = [s for s, _ in symbols[:top]]
    logger.info("Universe: %d symbols (min_vol=$%.0fM, top=%d)", len(result), min_volume / 1e6, top)
    return result


# ─── Klines ───────────────────────────────────────────────────────────────────

def fetch_klines(
    client: BinanceClient,
    symbol: str,
    interval: str = DEFAULT_KLINE_INTERVAL,
    days: int = KLINES_DAYS,
) -> pd.DataFrame:
    """Download OHLCV klines. Paginates automatically for long windows."""
    start_ms = _days_ago_ms(days)
    end_ms = _now_ms()
    all_rows = []

    while start_ms < end_ms:
        rows = client.get("/fapi/v1/klines", {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": _KLINE_LIMIT,
        })
        if not rows:
            break
        all_rows.extend(rows)
        # last candle open time + 1ms → next page
        start_ms = rows[-1][0] + 1
        if len(rows) < _KLINE_LIMIT:
            break

    if not all_rows:
        logger.warning("No klines for %s", symbol)
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = df[col].astype(float)
    df = df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]].copy()
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    df.drop_duplicates(inplace=True)

    # Quality filter: drop if fewer than MIN_KLINE_DAYS worth of rows
    min_rows = MIN_KLINE_DAYS * (24 // _interval_hours(interval))
    if len(df) < min_rows:
        logger.warning("%s: only %d candles (< %d min). Skipping.", symbol, len(df), min_rows)
        return pd.DataFrame()

    logger.debug("%s klines: %d rows [%s → %s]", symbol, len(df), df.index[0], df.index[-1])
    return df


def _interval_hours(interval: str) -> int:
    mapping = {"1h": 1, "2h": 2, "4h": 4, "6h": 6, "8h": 8, "12h": 12, "1d": 24}
    return mapping.get(interval, 4)


# ─── Funding Rate ─────────────────────────────────────────────────────────────

def fetch_funding_rate_history(
    client: BinanceClient,
    symbol: str,
    days: int = FR_DAYS,
) -> pd.DataFrame:
    """Download full funding rate history (up to 365 days)."""
    start_ms = _days_ago_ms(days)
    end_ms = _now_ms()
    all_rows = []

    while start_ms < end_ms:
        rows = client.get("/fapi/v1/fundingRate", {
            "symbol": symbol,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": _FR_LIMIT,
        })
        if not rows:
            break
        all_rows.extend(rows)
        start_ms = rows[-1]["fundingTime"] + 1
        if len(rows) < _FR_LIMIT:
            break

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    df = df[["timestamp", "funding_rate"]].drop_duplicates("timestamp").sort_values("timestamp")
    df.set_index("timestamp", inplace=True)
    return df


# ─── Helpers for period-aware endpoints ──────────────────────────────────────

def _fetch_period_endpoint(
    client: BinanceClient,
    path: str,
    symbol: str,
    period: str,
    days: int,
) -> list:
    """
    Fetch period data (OI/L-S/Taker) for the last `days` days.

    Does NOT send startTime — Binance returns the most recent available data
    up to `limit` rows.  Passing startTime older than ~7-14 days returns a
    -1130 error because these endpoints have a shorter availability window than
    their documented "30 days" (in practice varies).  We fetch the latest page
    and filter rows to the requested window afterward.
    """
    rows = client.get(path, {
        "symbol": symbol,
        "period": period,
        "limit": _OI_LIMIT,
    })
    if not rows:
        return []
    cutoff_ms = _days_ago_ms(days)
    return [r for r in rows if r.get("timestamp", 0) >= cutoff_ms]


# ─── Open Interest ────────────────────────────────────────────────────────────

def fetch_open_interest_hist(
    client: BinanceClient,
    symbol: str,
    days: int = OI_DAYS,
    interval: str = "4h",
) -> pd.DataFrame:
    """Download open interest history (Binance hard limit: ~30 days)."""
    rows = _fetch_period_endpoint(client, "/futures/data/openInterestHist", symbol, interval, days)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["open_interest"] = df["sumOpenInterest"].astype(float)
    df["oi_value_usdt"] = df["sumOpenInterestValue"].astype(float)
    df = df[["timestamp", "open_interest", "oi_value_usdt"]].drop_duplicates("timestamp").sort_values("timestamp")
    df.set_index("timestamp", inplace=True)
    return df


# ─── Long/Short Account Ratio ─────────────────────────────────────────────────

def fetch_long_short_ratio(
    client: BinanceClient,
    symbol: str,
    days: int = LS_DAYS,
    interval: str = "4h",
) -> pd.DataFrame:
    """Download global long/short account ratio."""
    rows = _fetch_period_endpoint(client, "/futures/data/globalLongShortAccountRatio", symbol, interval, days)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["ls_account_ratio"] = df["longShortRatio"].astype(float)
    df["ls_long_pct"] = df["longAccount"].astype(float)
    df["ls_short_pct"] = df["shortAccount"].astype(float)
    df = df[["timestamp", "ls_account_ratio", "ls_long_pct", "ls_short_pct"]].drop_duplicates("timestamp").sort_values("timestamp")
    df.set_index("timestamp", inplace=True)
    return df


# ─── Top Trader L/S Position Ratio ───────────────────────────────────────────

def fetch_top_trader_ratio(
    client: BinanceClient,
    symbol: str,
    days: int = LS_DAYS,
    interval: str = "4h",
) -> pd.DataFrame:
    """Download top trader long/short position ratio (smart money proxy)."""
    rows = _fetch_period_endpoint(client, "/futures/data/topLongShortPositionRatio", symbol, interval, days)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["top_trader_ls_ratio"] = df["longShortRatio"].astype(float)
    df["top_trader_long_pct"] = df["longAccount"].astype(float)
    df["top_trader_short_pct"] = df["shortAccount"].astype(float)
    df = df[["timestamp", "top_trader_ls_ratio", "top_trader_long_pct", "top_trader_short_pct"]].drop_duplicates("timestamp").sort_values("timestamp")
    df.set_index("timestamp", inplace=True)
    return df


# ─── Taker Buy/Sell Volume ────────────────────────────────────────────────────

def fetch_taker_volume(
    client: BinanceClient,
    symbol: str,
    days: int = TAKER_DAYS,
    interval: str = "4h",
) -> pd.DataFrame:
    """Download taker buy/sell volume ratio."""
    rows = _fetch_period_endpoint(client, "/futures/data/takerlongshortRatio", symbol, interval, days)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col, alias in [("buySellRatio", "taker_bs_ratio"), ("buyVol", "taker_buy_vol"), ("sellVol", "taker_sell_vol")]:
        if col in df.columns:
            df[alias] = df[col].astype(float)
        else:
            df[alias] = float("nan")

    keep = [c for c in ["timestamp", "taker_bs_ratio", "taker_buy_vol", "taker_sell_vol"] if c in df.columns]
    df = df[keep].drop_duplicates("timestamp").sort_values("timestamp")
    df.set_index("timestamp", inplace=True)
    return df


# ─── Parallel fetch for one symbol ───────────────────────────────────────────

def fetch_all_for_symbol(
    client: BinanceClient,
    symbol: str,
    interval: str = DEFAULT_KLINE_INTERVAL,
) -> dict[str, pd.DataFrame]:
    """
    Fetch all data for a single symbol. Returns dict keyed by data type.
    Each endpoint is fetched independently — a failure on one (e.g. OI/L-S/Taker
    not available for a symbol) returns an empty DataFrame for that type only.
    """
    result: dict[str, pd.DataFrame] = {}
    fetchers = {
        "klines":        lambda: fetch_klines(client, symbol, interval),
        "funding_rate":  lambda: fetch_funding_rate_history(client, symbol),
        "open_interest": lambda: fetch_open_interest_hist(client, symbol, interval=interval),
        "long_short":    lambda: fetch_long_short_ratio(client, symbol, interval=interval),
        "top_trader":    lambda: fetch_top_trader_ratio(client, symbol, interval=interval),
        "taker":         lambda: fetch_taker_volume(client, symbol, interval=interval),
    }
    for dtype, fn in fetchers.items():
        try:
            result[dtype] = fn()
        except Exception as exc:
            logger.warning("%s: %s endpoint failed (%s) — using empty DataFrame", symbol, dtype, exc)
            result[dtype] = pd.DataFrame()
    return result


def fetch_all_symbols(
    symbols: list[str],
    interval: str = DEFAULT_KLINE_INTERVAL,
    max_workers: int = 4,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Parallel download for multiple symbols.
    Returns {symbol: {data_type: DataFrame}}.
    Uses a shared client with a semaphore to respect rate limits.
    """
    results: dict[str, dict[str, pd.DataFrame]] = {}

    def _fetch(sym: str) -> tuple[str, dict]:
        with BinanceClient() as client:
            return sym, fetch_all_for_symbol(client, sym, interval)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch, s): s for s in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                _, data = fut.result()
                results[sym] = data
                logger.info("Fetched %s (%d klines)", sym, len(data.get("klines", [])))
            except Exception as exc:
                logger.error("Failed to fetch %s: %s", sym, exc)
                results[sym] = {}

    return results

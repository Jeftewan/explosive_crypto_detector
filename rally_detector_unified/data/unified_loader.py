"""
Canonical data pipeline: loads all sources, resamples to hourly grid,
merges, forward-fills, joins cross-exchange FR, and caches as Parquet.
"""
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from .binance_client import BinanceClient
from .binance_fetcher import (
    get_perp_symbols,
    fetch_all_symbols,
    fetch_all_for_symbol,
    DEFAULT_KLINE_INTERVAL,
)
from .postgres_loader import load_cross_exchange_fr
from ..config import (
    CACHE_DIR,
    CACHE_TTL_HOURS,
    CACHE_COMPRESSION,
    FORWARD_FILL_LIMIT_HOURS,
    DEFAULT_TOP_SYMBOLS,
    MIN_VOLUME_USDT,
)

logger = logging.getLogger(__name__)

_CACHE_UNIFIED = CACHE_DIR / "unified_grid.parquet"


def _cache_path(prefix: str, symbol: str) -> Path:
    safe = symbol.replace("/", "_")
    return CACHE_DIR / f"{prefix}_{safe}.parquet"


def _is_fresh(path: Path, ttl_hours: int = CACHE_TTL_HOURS) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime) < timedelta(hours=ttl_hours)


def _save(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression=CACHE_COMPRESSION)


def _load(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


# ─── Per-symbol raw cache ─────────────────────────────────────────────────────

def load_symbol_raw(
    symbol: str,
    force_reload: bool = False,
    interval: str = DEFAULT_KLINE_INTERVAL,
) -> dict[str, pd.DataFrame]:
    """
    Returns raw DataFrames (keyed by data type) for one symbol,
    using per-type Parquet cache when fresh.
    """
    types = ["klines", "funding_rate", "open_interest", "long_short", "top_trader", "taker"]
    cached: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for t in types:
        p = _cache_path(f"binance_{t}", symbol)
        if not force_reload and _is_fresh(p):
            cached[t] = _load(p)
        else:
            missing.append(t)

    if missing:
        with BinanceClient() as client:
            fresh = fetch_all_for_symbol(client, symbol, interval)
        for t, df in fresh.items():
            if not df.empty:
                _save(df, _cache_path(f"binance_{t}", symbol))
            cached[t] = df

    return cached


# ─── Resample to hourly grid ──────────────────────────────────────────────────

def _resample_to_hourly(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """Upsample any frequency to 1h grid with forward-fill."""
    if df.empty:
        return df
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    hourly = df.resample("1h").last()
    if method == "ffill":
        hourly = hourly.ffill(limit=FORWARD_FILL_LIMIT_HOURS)
    return hourly


# ─── Merge all sources for a symbol ──────────────────────────────────────────

def build_symbol_grid(
    symbol: str,
    raw: dict[str, pd.DataFrame],
    cross_exchange_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge all data types onto a 1h grid anchored to klines.
    Returns a DataFrame indexed by UTC hour with all features.
    """
    klines = raw.get("klines", pd.DataFrame())
    if klines.empty:
        return pd.DataFrame()

    # Anchor on klines hourly grid
    base = _resample_to_hourly(klines)

    # Merge auxiliary series
    for dtype in ["funding_rate", "open_interest", "long_short", "top_trader", "taker"]:
        src = raw.get(dtype, pd.DataFrame())
        if src.empty:
            continue
        hourly = _resample_to_hourly(src)
        base = base.join(hourly, how="left", rsuffix=f"_{dtype}")

    # Join cross-exchange FR from Postgres via nearest-hour merge (±2h tolerance)
    if cross_exchange_df is not None and not cross_exchange_df.empty:
        sym_xfr = cross_exchange_df[cross_exchange_df["symbol"] == symbol].copy()
        if not sym_xfr.empty:
            sym_xfr = sym_xfr.set_index("captured_at").sort_index()
            sym_xfr = sym_xfr[["fr_cross_exchange_std", "fr_cross_exchange_range"]]
            sym_xfr_h = _resample_to_hourly(sym_xfr)
            base = base.join(sym_xfr_h, how="left")

    base["symbol"] = symbol
    base.index.name = "timestamp"
    return base


# ─── Full universe pipeline ───────────────────────────────────────────────────

def load_unified_grid(
    symbols: list[str] | None = None,
    force_reload: bool = False,
    interval: str = DEFAULT_KLINE_INTERVAL,
    top: int = DEFAULT_TOP_SYMBOLS,
    min_volume: float = MIN_VOLUME_USDT,
    skip_fetch: bool = False,
) -> pd.DataFrame:
    """
    Main entry point. Returns a single long-format DataFrame:
        columns: symbol + all features
        index: timestamp (UTC hourly)

    If skip_fetch=True, loads only from cache (useful for fast iteration).
    """
    # Attempt to load full cache first
    if not force_reload and _is_fresh(_CACHE_UNIFIED):
        logger.info("Loading unified grid from cache: %s", _CACHE_UNIFIED)
        return _load(_CACHE_UNIFIED)

    # Resolve symbol universe
    if symbols is None:
        if skip_fetch:
            raise ValueError("symbols must be provided when skip_fetch=True and no unified cache exists")
        with BinanceClient() as client:
            symbols = get_perp_symbols(client, min_volume=min_volume, top=top)

    # Load cross-exchange FR from Postgres (best-effort)
    cross_exchange_df = load_cross_exchange_fr()

    # Collect per-symbol grids
    frames: list[pd.DataFrame] = []
    for sym in symbols:
        raw = load_symbol_raw(sym, force_reload=force_reload, interval=interval)
        grid = build_symbol_grid(sym, raw, cross_exchange_df)
        if not grid.empty:
            frames.append(grid)
        else:
            logger.debug("Empty grid for %s, skipping", sym)

    if not frames:
        raise RuntimeError("No valid symbol data loaded.")

    unified = pd.concat(frames, axis=0)
    unified.sort_index(inplace=True)

    _save(unified, _CACHE_UNIFIED)
    logger.info(
        "Unified grid: %d rows × %d cols, %d symbols",
        len(unified), unified.shape[1], unified["symbol"].nunique()
    )
    return unified

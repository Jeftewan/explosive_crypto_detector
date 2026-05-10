"""
Loads cross-exchange funding rate dispersion from Postgres.
"""
import logging

import pandas as pd

from ..config import POSTGRES_DSN, CROSS_EXCHANGE_DAYS

logger = logging.getLogger(__name__)


def _get_engine():
    try:
        from sqlalchemy import create_engine
        return create_engine(POSTGRES_DSN, pool_pre_ping=True)
    except ImportError:
        raise ImportError("sqlalchemy is required for postgres_loader. Run: pip install sqlalchemy psycopg2-binary")


def load_cross_exchange_fr(window_days: int = CROSS_EXCHANGE_DAYS) -> pd.DataFrame:
    """
    For each (symbol, captured_at), compute std and range of FR across exchanges.

    Expected source table: funding_rate_snapshots
    Columns needed: symbol, exchange, rate, captured_at

    Returns DataFrame with columns:
        symbol, captured_at, fr_cross_exchange_std, fr_cross_exchange_range
    """
    query = f"""
        SELECT
            symbol,
            DATE_TRUNC('hour', captured_at) AS captured_at,
            STDDEV(rate)                    AS fr_cross_exchange_std,
            MAX(rate) - MIN(rate)           AS fr_cross_exchange_range,
            COUNT(DISTINCT exchange)        AS exchange_count
        FROM funding_rate_snapshots
        WHERE
            captured_at >= NOW() - INTERVAL '{window_days} days'
            AND rate IS NOT NULL
        GROUP BY symbol, DATE_TRUNC('hour', captured_at)
        HAVING COUNT(DISTINCT exchange) >= 2
        ORDER BY symbol, captured_at
    """
    try:
        engine = _get_engine()
        df = pd.read_sql(query, engine)
        df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True)
        df["fr_cross_exchange_std"] = df["fr_cross_exchange_std"].astype(float)
        df["fr_cross_exchange_range"] = df["fr_cross_exchange_range"].astype(float)
        logger.info(
            "Cross-exchange FR: %d rows, %d unique symbols",
            len(df), df["symbol"].nunique()
        )
        return df
    except Exception as exc:
        logger.warning("Could not load cross-exchange FR from Postgres: %s", exc)
        return pd.DataFrame(columns=["symbol", "captured_at", "fr_cross_exchange_std", "fr_cross_exchange_range"])

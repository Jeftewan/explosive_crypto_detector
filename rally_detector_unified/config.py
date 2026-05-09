"""Global configuration and constants."""
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"
REPORTS_DIR = BASE_DIR / "reports"
CACHE_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# ─── Binance API ──────────────────────────────────────────────────────────────
BINANCE_BASE_URL = "https://fapi.binance.com"
REQUEST_DELAY_MS = 300          # ms between requests
MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 1.5        # exponential backoff multiplier

# ─── Data windows ─────────────────────────────────────────────────────────────
KLINES_DAYS = 365
FR_DAYS = 365
OI_DAYS = 30                    # Binance API hard limit
LS_DAYS = 30
TAKER_DAYS = 30
CROSS_EXCHANGE_DAYS = 43        # Postgres window

# ─── Universe filters ─────────────────────────────────────────────────────────
MIN_VOLUME_USDT = 5_000_000     # $5M daily volume
MIN_KLINE_DAYS = 60             # at least 60 days of history
DEFAULT_TOP_SYMBOLS = 300
DEFAULT_KLINE_INTERVAL = "4h"

# ─── Cache TTL ────────────────────────────────────────────────────────────────
CACHE_TTL_HOURS = 24
CACHE_COMPRESSION = "snappy"

# ─── Walk-forward ─────────────────────────────────────────────────────────────
WF_EMBARGO_DAYS = 21            # max horizon is 504h ≈ 21 days
WF_FOLDS = 5
WF_MIN_TRAIN_DAYS = 150

# ─── Return targets ───────────────────────────────────────────────────────────
# (threshold_pct, horizon_hours)
TARGETS = [
    (30,   24),
    (50,   24),
    (100,  24),
    (50,   72),
    (100,  72),
    (200,  72),
    (100,  168),
    (200,  168),
    (500,  504),
]

# ─── Feature engineering ──────────────────────────────────────────────────────
FR_WINDOW_DAYS = 7
FR_ZSCORE_WINDOW = 14
FR_MOMENTUM_PERIODS = 4
BB_WINDOW = 20
BB_STD = 2.0
RSI_PERIOD = 14
VOLUME_ZSCORE_WINDOW = 14
OBV_SMOOTHING = 10
FORWARD_FILL_LIMIT_HOURS = 6    # max gap fill for 4h OI data

# ─── Postgres ─────────────────────────────────────────────────────────────────
# Connection string loaded from environment variable POSTGRES_DSN or fallback.
import os
POSTGRES_DSN = os.getenv(
    "POSTGRES_DSN",
    "postgresql://localhost:5432/arbitrage"
)

"""
HTTP client for Binance Futures API with connection pooling, exponential
backoff on 429/5xx, and configurable rate limiting.
"""
import time
import logging
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config import BINANCE_BASE_URL, REQUEST_DELAY_MS, MAX_RETRIES, RETRY_BACKOFF_BASE

logger = logging.getLogger(__name__)


class BinanceClient:
    def __init__(
        self,
        base_url: str = BINANCE_BASE_URL,
        request_delay_ms: int = REQUEST_DELAY_MS,
        max_retries: int = MAX_RETRIES,
    ):
        self.base_url = base_url.rstrip("/")
        self._delay = request_delay_ms / 1000.0
        self._max_retries = max_retries
        self._session = self._build_session()

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=self._max_retries,
            backoff_factor=RETRY_BACKOFF_BASE,
            status_forcelist=[429, 500, 502, 503, 504],
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=20)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def get(self, path: str, params: dict | None = None) -> Any:
        url = f"{self.base_url}{path}"
        logger.debug("GET %s params=%s", url, params)
        time.sleep(self._delay)
        for attempt in range(1, self._max_retries + 1):
            resp = self._session.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", RETRY_BACKOFF_BASE ** attempt))
                logger.warning("Rate limited (429). Waiting %.1fs (attempt %d)", wait, attempt)
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                wait = RETRY_BACKOFF_BASE ** attempt
                logger.warning("Server error %d. Waiting %.1fs (attempt %d)", resp.status_code, wait, attempt)
                time.sleep(wait)
                continue
            if resp.status_code == 400:
                # Log Binance error body (e.g. {"code":-1128,"msg":"..."}) before raising
                try:
                    body = resp.json()
                    logger.error("Binance 400 — code=%s msg=%s | %s params=%s",
                                 body.get("code"), body.get("msg"), path, params)
                except Exception:
                    logger.error("Binance 400 — %s | %s params=%s", resp.text[:200], path, params)
                resp.raise_for_status()
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"Max retries exceeded for {url}")

    def close(self):
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

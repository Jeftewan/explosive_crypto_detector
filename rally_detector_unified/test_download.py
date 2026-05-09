"""
Quick connectivity and data download test using BTC, ETH, SOL.
Run before the full backtest to verify the data pipeline works.

Usage:
    python -m rally_detector_unified.test_download
    python -m rally_detector_unified.test_download --verbose
"""
import argparse
import logging
import sys

from .data.binance_client import BinanceClient
from .data.binance_fetcher import (
    fetch_klines,
    fetch_funding_rate_history,
    fetch_open_interest_hist,
    fetch_long_short_ratio,
    fetch_top_trader_ratio,
    fetch_taker_volume,
    get_perp_symbols,
)
from .data.unified_loader import load_symbol_raw, build_symbol_grid
from .backtest.feature_builder import build_features_for_symbol

TEST_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def run_test(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s — %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("test_download")

    log.info("=== Test Download: %s ===", TEST_SYMBOLS)
    passed = 0
    failed = 0

    with BinanceClient() as client:

        # ── Test 1: Symbol universe ───────────────────────────────────────────
        log.info("[1/7] Getting universe of perp symbols...")
        try:
            syms = get_perp_symbols(client, top=10)
            assert len(syms) > 0, "Empty symbol list"
            log.info("  ✓ Got %d symbols. Top 5: %s", len(syms), syms[:5])
            passed += 1
        except Exception as e:
            log.error("  ✗ get_perp_symbols FAILED: %s", e)
            failed += 1

        # ── Test 2-7: Per-symbol endpoints ────────────────────────────────────
        sym = "BTCUSDT"

        for name, fn, kwargs in [
            ("klines (4h, 7d)", fetch_klines, {"days": 7}),
            ("funding_rate (7d)", fetch_funding_rate_history, {"days": 7}),
            ("open_interest (7d)", fetch_open_interest_hist, {"days": 7}),
            ("long_short_ratio (7d)", fetch_long_short_ratio, {"days": 7}),
            ("top_trader_ratio (7d)", fetch_top_trader_ratio, {"days": 7}),
            ("taker_volume (7d)", fetch_taker_volume, {"days": 7}),
        ]:
            log.info("[%d/7] %s — %s...", passed + failed + 2, sym, name)
            try:
                df = fn(client, sym, **kwargs)
                if df.empty:
                    log.warning("  ⚠️  Empty DataFrame for %s %s (may be normal for some symbols)", sym, name)
                    passed += 1
                else:
                    log.info("  ✓ %d rows [%s → %s]", len(df), df.index[0].date(), df.index[-1].date())
                    passed += 1
            except Exception as e:
                log.error("  ✗ %s FAILED: %s", name, e)
                failed += 1

    # ── Test: unified loader for 3 symbols ────────────────────────────────────
    log.info("\n[Full pipeline] Loading 3-symbol grid...")
    try:
        for sym in TEST_SYMBOLS:
            raw = load_symbol_raw(sym, force_reload=False)
            grid = build_symbol_grid(sym, raw)
            if grid.empty:
                log.warning("  ⚠️  Empty grid for %s", sym)
            else:
                log.info("  ✓ %s: %d rows × %d cols", sym, len(grid), grid.shape[1])

            feat = build_features_for_symbol(grid)
            if feat.empty:
                log.warning("  ⚠️  No features for %s", sym)
            else:
                target_cols = [c for c in feat.columns if c.startswith("rally_")]
                log.info("    Features: %d cols, %d targets", feat.shape[1], len(target_cols))
                for tc in target_cols:
                    pos = feat[tc].mean()
                    log.info("      %s: %.2f%% positive", tc, pos * 100)
        passed += 1
    except Exception as e:
        log.error("  ✗ Full pipeline FAILED: %s", e)
        import traceback
        traceback.print_exc()
        failed += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    total = passed + failed
    log.info("\n=== Results: %d/%d passed ===", passed, total)
    if failed == 0:
        log.info("✅ All tests passed. Ready for full backtest.")
        log.info("\nNext step:")
        log.info("  python -m rally_detector_unified.main --top 300")
    else:
        log.error("❌ %d test(s) failed. Fix before running full backtest.", failed)

    return 0 if failed == 0 else 1


def main():
    parser = argparse.ArgumentParser(description="Test Binance API connectivity and data pipeline")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    sys.exit(run_test(verbose=args.verbose))


if __name__ == "__main__":
    main()

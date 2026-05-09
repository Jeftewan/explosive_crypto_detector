"""
Main orchestrator for the unified rally detector.

Usage:
    python -m rally_detector_unified.main
    python -m rally_detector_unified.main --top 300 --force-reload
    python -m rally_detector_unified.main --symbols BTCUSDT ETHUSDT SOLUSDT
    python -m rally_detector_unified.main --skip-fetch --horizons 24h,72h,168h
    python -m rally_detector_unified.main --verbose
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .data.unified_loader import load_unified_grid
from .data.binance_fetcher import fetch_klines
from .data.binance_client import BinanceClient
from .backtest.feature_builder import build_features_universe
from .backtest.walk_forward import run_walk_forward
from .backtest.ground_truth import validate_against_user_history
from .scoring.logistic_l1 import train_all_models, predict_proba_all
from .analysis.score_buckets import score_buckets_analysis
from .analysis.multi_profile import multi_profile_analysis
from .analysis.pre_explosion import pre_explosion_analysis
from .analysis.risk_reward import simulate_equal_weight, outlier_contribution_analysis
from .analysis.top_explosions import top_explosions_analysis
from .analysis.correlations import correlation_analysis
from .analysis.walk_forward_stability import walk_forward_stability_analysis
from .analysis.feature_importance import feature_importance_analysis
from .analysis.optimal_holding import optimal_holding_analysis
from .analysis.market_regime import market_regime_breakdown
from .analysis.rally_type_breakdown import rally_type_breakdown
from .reports.markdown_writer import generate_report
from .config import DEFAULT_TOP_SYMBOLS, MIN_VOLUME_USDT, TARGETS, REPORTS_DIR


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Unified crypto rally detector with walk-forward backtesting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_SYMBOLS,
                        help="Top N symbols by 24h volume")
    parser.add_argument("--min-volume", type=float, default=MIN_VOLUME_USDT,
                        help="Minimum 24h USDT volume filter")
    parser.add_argument("--force-reload", action="store_true",
                        help="Ignore Parquet cache and re-download everything")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip Binance download, use only cached data")
    parser.add_argument("--symbols", nargs="+", metavar="SYM",
                        help="Only analyze specific symbols (e.g. BTCUSDT ETHUSDT)")
    parser.add_argument("--horizons", type=str, default=None,
                        help="Comma-separated horizon filter e.g. '24h,72h,168h'")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Backtest start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None,
                        help="Backtest end date YYYY-MM-DD")
    parser.add_argument("--max-days", type=int, default=None,
                        help="Limit backtest to last N days (reduces memory for large universes)")
    parser.add_argument("--no-html", action="store_true",
                        help="Skip generating the HTML dashboard")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable DEBUG logging")
    return parser.parse_args(argv)


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv=None):
    args = parse_args(argv)
    setup_logging(args.verbose)
    log = logging.getLogger("main")

    log.info("=== Rally Detector Unified — starting ===")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    log.info("Phase 1/5: Loading data...")
    unified_df = load_unified_grid(
        symbols=args.symbols,
        force_reload=args.force_reload,
        top=args.top,
        min_volume=args.min_volume,
        skip_fetch=args.skip_fetch,
    )

    # Optional date slice
    if args.max_days:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=args.max_days)
        unified_df = unified_df[unified_df.index >= cutoff]
        log.info("Limiting to last %d days (from %s)", args.max_days, cutoff.date())
    if args.start_date:
        unified_df = unified_df[unified_df.index >= pd.Timestamp(args.start_date, tz="UTC")]
    if args.end_date:
        unified_df = unified_df[unified_df.index <= pd.Timestamp(args.end_date, tz="UTC")]

    log.info("Loaded %d rows for %d symbols", len(unified_df), unified_df["symbol"].nunique())

    # ── 2. Build features ─────────────────────────────────────────────────────
    log.info("Phase 2/5: Building features & targets...")

    # BTC data for market regime
    btc_df = pd.DataFrame()
    btc_sym = "BTCUSDT"
    if btc_sym in unified_df["symbol"].values:
        btc_df = unified_df[unified_df["symbol"] == btc_sym][["open", "high", "low", "close", "volume"]].copy()

    feature_df = build_features_universe(unified_df, btc_df=btc_df if not btc_df.empty else None)
    log.info("Feature matrix: %d rows × %d cols", len(feature_df), feature_df.shape[1])

    # ── 3. Walk-forward CV ────────────────────────────────────────────────────
    log.info("Phase 3/5: Walk-forward cross-validation...")
    wf_kwargs: dict = {}
    if args.max_days and args.max_days < 180:
        # Scale down walk-forward params proportionally for short windows
        wf_kwargs = {
            "min_train_days": max(30, args.max_days // 2),
            "n_folds": 2 if args.max_days < 120 else 3,
        }
        log.info("Short window detected: using min_train_days=%d, n_folds=%d",
                 wf_kwargs["min_train_days"], wf_kwargs["n_folds"])
    fold_results = run_walk_forward(feature_df, **wf_kwargs)

    # ── 4. Final model on full data ───────────────────────────────────────────
    log.info("Phase 4/5: Training final model on full data...")
    final_models = train_all_models(feature_df)
    predictions = predict_proba_all(feature_df, final_models)

    # ── 5. All analyses ───────────────────────────────────────────────────────
    log.info("Phase 5/5: Running analyses...")

    primary_target = "rally_100_72h"

    target_cols = [f"rally_{t}_{h}h" for t, h in TARGETS if f"rally_{t}_{h}h" in feature_df.columns]
    actuals = feature_df[target_cols]

    stability = walk_forward_stability_analysis(fold_results, target_col=primary_target)
    buckets = score_buckets_analysis(predictions, actuals)
    profiles = multi_profile_analysis(feature_df, actuals)
    footprint = pre_explosion_analysis(feature_df, target_col=primary_target)
    rr = simulate_equal_weight(feature_df, predictions, target_col=primary_target)
    outliers = outlier_contribution_analysis(feature_df, target_col=primary_target)
    top_exp = top_explosions_analysis(feature_df, target_col=primary_target)
    corrs = correlation_analysis(feature_df, target_col=primary_target)
    fi = feature_importance_analysis(final_models)
    holding = optimal_holding_analysis(feature_df)
    regime = market_regime_breakdown(feature_df, predictions)
    rtypes = rally_type_breakdown(feature_df, target_col=primary_target)
    gt = validate_against_user_history(predictions, feature_df)

    # ── 6. Report ─────────────────────────────────────────────────────────────
    log.info("Writing report...")
    report_md = generate_report(
        fold_results=fold_results,
        stability=stability,
        score_buckets=buckets,
        multi_profile=profiles,
        pre_explosion=footprint,
        risk_reward=rr,
        top_explosions=top_exp,
        correlations=corrs,
        feature_importance=fi,
        optimal_holding=holding,
        regime_breakdown=regime,
        rally_types=rtypes,
        ground_truth=gt,
    )

    if not args.no_html:
        try:
            from .reports.html_dashboard import generate_html_dashboard
            generate_html_dashboard(
                fold_metrics=stability.get("fold_metrics", pd.DataFrame()),
                score_buckets=buckets,
                feature_importance=fi,
                multi_profile=profiles,
            )
        except ImportError:
            log.warning("plotly not installed — skipping HTML dashboard (pip install plotly)")

    log.info("=== Done! Reports in: %s ===", REPORTS_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())

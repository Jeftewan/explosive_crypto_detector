# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Idioma

El usuario trabaja en español. Responde en español por defecto. Mantén el código, identificadores, docstrings y commits en inglés (estándar del proyecto).

## Comandos comunes

Todo se ejecuta como módulo desde la raíz del repo (`explosive_crypto_detector/`), porque el paquete usa imports relativos (`from .data...`).

```bash
# Smoke test rápido (3 símbolos, ~30s) — correr SIEMPRE antes del backtest completo
python -m rally_detector_unified.test_download --verbose

# Backtest completo (descarga inicial 30-60 min, luego segundos desde caché)
python -m rally_detector_unified.main --top 300

# Iteración rápida sin re-descargar
python -m rally_detector_unified.main --skip-fetch --symbols BTCUSDT ETHUSDT SOLUSDT

# Forzar re-descarga (ignora caché Parquet)
python -m rally_detector_unified.main --force-reload

# Instalación de deps
pip install -r rally_detector_unified/requirements.txt
```

No hay suite de tests unitarios ni linter configurado. `test_download.py` es un smoke test de conectividad/pipeline, no un test unitario — cuenta endpoints que devuelven `df.empty` como "pass".

## Arquitectura — visión global

Pipeline de 5 fases orquestado por `rally_detector_unified/main.py`. Datos crudos → grid horario unificado → features+targets → walk-forward CV → modelo final + 11 análisis → reporte Markdown/HTML.

**Flujo de datos:**

```
Binance Futures API (público)         Postgres (opcional)
  ├─ klines (365d)                      ├─ funding_rate_snapshots
  ├─ fundingRate (365d)                 │   → cross-exchange dispersion
  ├─ openInterestHist (30d HARD LIMIT)  └─ user_history (13 trades, sanity check)
  ├─ globalLongShortAccountRatio (30d)
  ├─ topLongShortPositionRatio (30d)
  └─ takerlongshortRatio (30d)
              │
              ▼
data/unified_loader.py  ──►  resample 4h→1h, ffill ≤6h, merge por timestamp
              │              cache parquet por símbolo + un unified_grid.parquet (TTL 24h)
              ▼
backtest/feature_builder.py ──►  ~28 features (indicators/) + 9 targets rally_{pct}_{h}h
              │
              ▼
backtest/walk_forward.py    ──►  5 folds purged + embargo 21d (López de Prado)
              │
              ▼
scoring/logistic_l1.py      ──►  9 modelos LogisticRegressionCV L1 (saga, balanced)
              │              uno por target — ver TARGETS en config.py
              ▼
analysis/* (11 módulos)     ──►  reports/markdown_writer.py + html_dashboard.py
```

**Detalles que requieren leer varios archivos:**

- **Grid horario es el ancla**. `unified_loader._resample_to_hourly` upsamplea todo a 1h con `ffill(limit=6)`. Las velas de 4h se repiten 4 veces; OI/L-S/Taker (también 4h en API) se rellenan igual. Los huecos >6h quedan NaN — el imputer del pipeline (`SimpleImputer(strategy="median", add_indicator=True)`) los maneja añadiendo columna indicadora.
- **Asimetría de cobertura**: klines y FR cubren 365 días; OI/L-S/Taker solo los últimos ~30 días por límite de la API. Los folds 1-3 del walk-forward entrenan/testean en períodos sin esos derivados → siempre NaN. Esto es esperado, no un bug.
- **Targets son look-ahead-safe**. `feature_builder._compute_targets` usa `close.rolling(window=horizon).max().shift(-horizon)` y enmascara las últimas `horizon` filas como NaN. No tocar sin entender la fórmula del docstring.
- **Universo se filtra dos veces**: `get_perp_symbols` filtra por volumen 24h ≥ $5M (`MIN_VOLUME_USDT`), y `fetch_klines` descarta símbolos con < `MIN_KLINE_DAYS=60` días (= 360 velas 4h). El warning "only N candles (< 360 min). Skipping" es ese segundo filtro disparándose.
- **Postgres es best-effort**. `postgres_loader` envuelve todo en try/except y devuelve DataFrames vacíos si falla la conexión. El pipeline sigue funcionando sin Postgres — solo pierde features cross-exchange y validación contra `user_history`.

## Convenciones del proyecto

- **Toda configuración vive en `config.py`** — paths, ventanas, thresholds, hiperparámetros del CV, lista `TARGETS`. Si necesitas un valor mágico, primero busca ahí.
- **Imports relativos** (`from .data.binance_client import ...`). Por eso siempre se ejecuta con `python -m rally_detector_unified.X`, nunca `python rally_detector_unified/X.py`.
- **Caché Parquet en `rally_detector_unified/cache/`** con TTL 24h. Borra el directorio si quieres re-fetch limpio sin pasar `--force-reload`.
- **DatetimeIndex siempre UTC con tz-aware**. Cualquier merge de timestamps debe respetar esto o reventará con error de tz mismatch.
- **Logging con `logging.getLogger(__name__)`** en cada módulo. `--verbose` activa DEBUG global. No uses `print`.

## Trampas conocidas / contexto del usuario

- **El sistema corre en Windows** (`C:\Users\jefte\OneDrive\Documentos\Arbitraje Cripto\Rally_detector`). Cuidado con paths hardcoded y separadores.
- **Rate limiting Binance**: el cliente espera 300ms entre requests (`REQUEST_DELAY_MS`) y reintenta 5× con backoff exponencial. No bajes esto sin probar — `-1130` y `429` aparecen rápido si paralelizas en exceso.
- **Targets `rally_*` con thresholds 30-500%** están calibrados para small/mid-caps que pueden hacer pumps explosivos. En BTC/ETH/SOL casi todos los targets dan 0% positivos en 1 año — esto es esperado, no un bug del feature builder. El smoke test no valida la lógica de target labelling.
- **`OneDrive` puede bloquear archivos** del directorio `cache/` mientras sincroniza. Si ves `PermissionError` al escribir parquet, es OneDrive, no el código.
- **Plan original detallado en `PLAN_UNIFICADO_v3_FINAL.md`** (raíz del repo). Es la spec de diseño y el contrato de las 8 fases. Consúltalo si una decisión arquitectónica no es obvia desde el código.

## Trabajo en este repo

- **Branch de desarrollo**: `claude/test-crypto-download-JGpxO`. Commitear y pushear ahí salvo instrucción explícita en contrario.
- **No crear PRs** sin pedido explícito del usuario.
- Cuando edites indicadores en `indicators/`, considera si afecta la lista `expected features` que `scoring/feature_pipeline.py:build_feature_matrix` selecciona. Añadir feature implica registrarla allí.
- Antes de cambios grandes en pipeline de datos, valida con `python -m rally_detector_unified.test_download --verbose` — corre en ~30s y cubre los 6 endpoints + el grid completo para 3 símbolos.

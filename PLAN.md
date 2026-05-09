Plan Unificado v3 — Detector de Rallies Cripto Multi-Factor (FINAL)
Versión: 3.0 — APROBADO Y LISTO PARA IMPLEMENTACIÓN
Fecha: 2026-05-09
Modelo predictivo: Regresión logística con regularización L1 (Lasso)
Fuente primaria de datos: Binance Futures API (pública)
Fuente secundaria: Postgres (solo para dispersión cross-exchange)
Plataforma: Windows local (C:/Users/jefte/OneDrive/Documentos/Arbitraje Cripto/Fouding Bot)

Decisión arquitectónica final: Binance primario, Postgres complementario
Tras analizar el tradeoff, la arquitectura final es:
DatoFuenteHistóricoPrecio (OHLCV)Binance klines365 díasVolumen real por períodoBinance klines365 díasFunding rateBinance fundingRate365 díasOpen InterestBinance openInterestHist30 días (límite API)Long/Short Account RatioBinance globalLongShortAccountRatio30 díasTaker Buy/Sell VolumeBinance takerlongshortRatio30 díasTop Trader L/S Position RatioBinance topLongShortPositionRatio30 díasFR Cross-Exchange DispersionPostgres (única fuente)43 díasValidación con trades realesPostgres user_history13 trades
Razón: Binance da 365 días de histórico vs 43 de Postgres → 8.5× más datos = walk-forward con 5-7 folds = conclusiones estadísticamente sólidas. Postgres aporta valor único en cross-exchange y trades reales.

Hallazgos críticos confirmados

open_interest 100% NULL en Postgres → ese indicador en tu detector actual estaba operando con ruido. Usaremos OI de Binance.
Tu detector actual mide retornos en "intervalos" (8iv = 8 cobros de FR), pero los intervalos varían por moneda (1h/4h/8h). Esto compara peras con manzanas. Usaremos horas reales (24h/72h/168h/504h) sobre klines de Binance.
Composite score con pesos hardcoded (score += 12, score += 8...) no es óptimo y no se adapta a la data. Reemplazado por regresión logística L1.
Sin walk-forward con purging → tu backtest actual tiene riesgo alto de overfitting. Implementamos esquema de López de Prado.


Hallazgos de la investigación académica (5 referencias clave)
ReferenciaAporte aplicadoBIS WP 1087 — "Crypto carry"Basis premium (perp − spot) más limpio que FR soloFieberg et al. 2025 (MDPI)CEX dominan price discovery; FR cross-exchange dispersion como señalLópez de Prado 2018Walk-forward con purging + embargoArian, Norouzi & Seco 2024 (SSRN)Métricas PBO y DSR para detectar overfittingarXiv:2412.18848 (2024)Combinar trade + order book → +10pp accuracy

Las 8 fases del plan (revisadas y finales)
FASE 1 — Inspección de DB ✅ COMPLETADA
Resultado documentado. OI null, multi-exchange disponible, 43 días en Postgres, user_history con 13 trades reales.

FASE 2 — Data layer (PRIMERA EN IMPLEMENTAR)
Estructura de directorios dentro de tu carpeta del bot:
C:/Users/jefte/OneDrive/Documentos/Arbitraje Cripto/Fouding Bot/
└── rally_detector_unified/                 # ← carpeta nueva
    ├── data/
    │   ├── binance_client.py               # Cliente HTTP con retry
    │   ├── binance_fetcher.py              # Klines + FR + OI + L/S + Taker
    │   ├── postgres_loader.py              # Solo cross-exchange + user_history
    │   └── unified_loader.py               # Combina, resamplea, cachea
    ├── cache/                              # Parquet local
    │   ├── binance_klines_<symbol>.parquet
    │   ├── binance_fr_<symbol>.parquet
    │   ├── binance_oi_<symbol>.parquet
    │   ├── binance_ls_<symbol>.parquet
    │   ├── binance_taker_<symbol>.parquet
    │   └── unified_grid.parquet
    └── ...
Tareas técnicas:

binance_client.py — wrapper con:

Pool de sesiones HTTP.
Retry exponencial en 429 y 5xx.
Rate limiting respetuoso (300ms entre requests, configurable).
Logging de cada request.


binance_fetcher.py — descargas paralelas (con semáforo para no exceder rate limit):

get_perp_symbols(): lista de USDT-perpetuos activos. Filtra inactivos.
fetch_klines(symbol, interval, days): hasta 365 días, intervalo 4h por defecto. Pagina con startTime/endTime.
fetch_funding_rate_history(symbol, days): hasta 365 días.
fetch_open_interest_hist(symbol, days): 30 días, intervalo 4h.
fetch_long_short_ratio(symbol, days): 30 días, account ratio.
fetch_top_trader_ratio(symbol, days): 30 días, top trader position ratio (señal "smart money").
fetch_taker_volume(symbol, days): 30 días.


postgres_loader.py — solo lo necesario:

load_cross_exchange_fr(window_days): para cada (symbol, captured_at), calcula la desviación estándar del FR entre exchanges y la dispersión max-min. Output: DataFrame con [symbol, captured_at, fr_cross_exchange_std, fr_cross_exchange_range].
load_user_history(): 13 trades para validación posterior.


unified_loader.py — pipeline canónico:

Carga klines de Binance → DataFrame con grid horario (interval 4h resampleado a 1h con forward-fill).
Hace lo mismo con FR, OI, L/S, taker.
Merge por (symbol, timestamp_horario) alineando todos los datasets.
Forward-fill ≤6h (porque OI viene cada 4h), NaN si más.
Joinea cross-exchange dispersion de Postgres por (symbol, hora más cercana ±2h).
Output: un solo DataFrame con todas las features brutas listas para feature engineering.


Filtros de calidad:

Descartar símbolos con <60 días de klines en Binance.
Descartar mark_price <= 0 (en realidad Binance no debería tenerlos, pero por seguridad).
Descartar pares con volumen 24h < $5M (configurable con --min-volume).
Top 300-500 símbolos por volumen (configurable con --top).


Caché Parquet:

TTL 24h (configurable).
--force-reload ignora caché.
Compresión snappy para velocidad.



Tiempo estimado de descarga inicial: ~30-60 minutos para 300 símbolos × 5 endpoints. Posteriormente cargas en segundos desde caché.
Limitación documentada: OI/L-S/Taker solo cubren los últimos 30 días del backtest. Para el resto del histórico (335 días si descargamos 365), esos indicadores son NaN y el modelo logístico los maneja como categoría aparte.

FASE 3 — Catálogo de indicadores ✅ APROBADO
MANTENER (12 indicadores)
IndicadorImplementaciónFR % positivo (ventana)Ventana de 7 días, sin cambiosFR racha consecutivaSin cambiosFR z-scoreSobre ventana rolling 14dFR momentum (ROC 4 períodos)Sin cambiosFR percentilSobre ventana rolling 14dBB Squeeze + squeeze_barsSobre klines de 4hBB %BSin cambiosVolume z-score / spikeSobre klines de 4hRSI(14)Sin cambiosPrice ROC 7/14/21dSin cambiosOBV trend / accumulationSin cambiosVolatility compressionSin cambios
MODIFICAR
IndicadorCambioHorizontes de retorno"8iv/24iv/72iv" → "24h/72h/168h/504h"OI z-score / OI regimeOI de Binance, no de Postgres. NaN-awareComposite scoreReemplazado por regresión logística L1RSI sweet spotNo asumir 25-45 → backtest descubre rango óptimoRSI divergenceAuditar look-ahead bias en sub-ventanas
AÑADIR (8 indicadores nuevos)
IndicadorCálculoJustificaciónLong/Short Account RatioDirecto de BinancePosicionamiento minoristaL/S Ratio z-scoreSobre ventana rolling 7dDetecta extremosTop Trader L/S Position RatioDirecto de BinanceSmart money positioningL/S Divergenceaccount_ratio vs top_trader_ratioCuando minorista vs profesional opinan distintoTaker Buy/Sell RatiobuyVol / sellVolPresión compradora agresivaTaker B/S momentumROC del ratio sobre 8 períodosCambio de direcciónBTC market regimeBTC > MA200 (binario)Filtro bull/bearFR Cross-Exchange Dispersionstd de FR entre exchanges (Postgres)Tensión arbitrajeVolume rank percentilRank del símbolo en universoLiquidez relativa
ELIMINAR
IndicadorRazónfr_contrarian (= -1 × rate × 1000)Redundante con FR z-scorefr_quintil con thresholds hardcodedReemplazado por z-score continuoComposite score heurístico (score += N)Reemplazado por regresión logística L1
Total features finales: ~28 (12 mantenidos + 8 añadidos + 8 derivados/transformados).

FASE 4 — Esqueleto del script unificado
rally_detector_unified/
├── config.py                       # Constantes globales y paths
├── main.py                         # Orquestador con argparse
├── data/
│   ├── binance_client.py
│   ├── binance_fetcher.py
│   ├── postgres_loader.py
│   └── unified_loader.py
├── indicators/
│   ├── fr.py                       # FR indicators (mantenidos)
│   ├── technical.py                # RSI, BB, Volume, OBV, momentum
│   ├── derivs.py                   # OI, L/S, Top Trader, Taker
│   ├── regime.py                   # BTC market regime + dominance
│   └── cross_exchange.py           # FR dispersion entre exchanges
├── scoring/
│   ├── feature_pipeline.py         # Estandarización + NaN handling
│   └── logistic_l1.py              # Regresión logística L1
├── backtest/
│   ├── feature_builder.py          # build_features con grid horario
│   ├── walk_forward.py             # Purged CV con embargo
│   ├── metrics.py                  # Hit rate, Kelly, Sharpe, PBO, DSR, max DD
│   └── ground_truth.py             # Validar contra user_history
├── analysis/
│   ├── score_buckets.py
│   ├── multi_profile.py
│   ├── pre_explosion.py
│   ├── risk_reward.py              # Aquí va "1 de 10 sube x20"
│   ├── top_explosions.py
│   ├── correlations.py
│   ├── walk_forward_stability.py
│   ├── feature_importance.py
│   ├── optimal_holding.py
│   ├── market_regime.py
│   └── rally_type_breakdown.py     # Tipo A/B/C
├── reports/
│   ├── markdown_writer.py
│   └── html_dashboard.py           # Plotly opcional
├── cache/                          # Parquet local
├── reports/                        # Outputs
└── README.md
Argumentos principales del script:
bash# Backtest completo con defaults sensatos
python main.py

# Top 300 símbolos por volumen
python main.py --top 300

# Ignorar caché y descargar todo de nuevo
python main.py --force-reload

# Saltar descarga de Binance (usa caché)
python main.py --skip-fetch

# Solo ciertos horizontes
python main.py --horizons 24h,72h,168h

# Filtro de liquidez
python main.py --min-volume 5e6

# Ventana específica
python main.py --start-date 2025-05-09 --end-date 2026-05-09

# Modo verbose
python main.py --verbose

# Solo símbolos específicos (debug)
python main.py --symbols BTCUSDT ETHUSDT SOLUSDT

FASE 5 — Walk-forward con purging (mejorado por más datos)
Con 365 días disponibles (Binance klines), el esquema clásico de López de Prado funciona bien:
Datos: 365 días totales
─────────────────────────────────────────────────────────
Fold 1:  [Train: días 0-150]    [Embargo: 21d]  [Test: días 171-200]
Fold 2:  [Train: días 0-200]    [Embargo: 21d]  [Test: días 221-250]
Fold 3:  [Train: días 0-250]    [Embargo: 21d]  [Test: días 271-300]
Fold 4:  [Train: días 0-300]    [Embargo: 21d]  [Test: días 321-350]
Fold 5:  [Train: días 0-350]    [Embargo: 21d]  [Test: días 350-365]  (último, más corto)
─────────────────────────────────────────────────────────
Embargo: 21 días (≈ horizonte máximo 504h) para que no haya leakage por autocorrelación.
Purging: se eliminan del train las observaciones cuyo retorno futuro se solapa con el período de test.
Métricas por fold:

Hit rate ≥30/50/100/200% por horizonte.
Retorno medio + mediana + desviación.
Sharpe anualizado.
Max drawdown intra-trade.
Kelly ajustado por drawdown.
PBO (probability of backtest overfitting).
DSR (Deflated Sharpe Ratio).

Excepción: el indicador OI / L-S / Taker solo cubren los últimos 30 días → en folds 1-3 (que terminan antes de los últimos 30 días) esos indicadores son siempre NaN. El modelo logístico los maneja, pero reportaremos explícitamente que el modelo se entrena con dos regímenes de cobertura (con y sin derivados auxiliares) y separaremos resultados.

FASE 6 — Regresión logística L1 ✅ ELEGIDO
Pipeline:
pythonfrom sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    # NaN → mediana del train (NaN como valor extra para indicadores OI fuera de rango)
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    # Estandarización: mean=0, std=1
    ("scaler", StandardScaler()),
    # Logistic con Lasso
    ("model",  LogisticRegressionCV(
        Cs=10,
        cv=5,                                # CV interno solo para hyperparam
        penalty="l1",
        solver="saga",
        max_iter=2000,
        scoring="average_precision",
        class_weight="balanced",
        n_jobs=-1,
    )),
])
Targets entrenados (uno por horizonte+threshold):
TargetDefiniciónrally_30_24h¿subió ≥30% en 24h?rally_50_24h¿subió ≥50% en 24h?rally_100_24h¿subió ≥100% en 24h? (raro, alta señal)rally_50_72h≥50% en 72hrally_100_72h≥100% en 72h (probablemente el más útil)rally_200_72h≥200% en 72hrally_100_168h≥100% en 7 díasrally_200_168h≥200% en 7 díasrally_500_504h≥500% en 21 días (lottery ticket)
Cada uno produce su propio modelo entrenado y sus propias probabilidades. Total: 9 modelos.
Métricas de éxito por modelo:

Average Precision (AP) > 2× clase base.
Calibración: en bucket de proba 70-80%, la frecuencia real debe estar 70-80% (Brier score, ECE).
Top decil: hit rate ≥3× el baseline.
Coeficientes esparsos: idealmente solo 5-10 features con coef ≠ 0 (señal de que el modelo está aprendiendo, no memorizando).


FASE 7 — Análisis y reporte (11 análisis, 1 especial)
Mantenidos del actual (mejorados):

Score buckets — probabilidad logística por bucket vs hit rate real (calibración visual).
Perfiles multi-indicador — combinaciones (BB Squeeze + Vol spike + RSI < 50, etc.).
Huella pre-explosión — qué indicadores tenían los tokens que sí explotaron.
Risk/reward simulación — capital igual por señal, holding fijo.
Top explosiones — los que más subieron + sus indicadores.
Correlaciones — Pearson + Spearman.

Nuevos:

Walk-forward stability — Sharpe / hit rate / Kelly por fold. Detecta degradación temporal.
Feature importance — coeficientes Lasso ordenados por magnitud absoluta.
Optimal holding por señal — para cada perfil, expectativa por horizonte. Te dice qué horizonte (24h/72h/168h/504h) maximiza retorno por señal.
Market regime breakdown — hit rate y Sharpe en BTC bull (BTC > MA200) vs bear.
Rally type breakdown (A/B/C):

Tipo A (acumulación silenciosa): BB Squeeze + OBV positivo + FR neutro.
Tipo B (short squeeze): OI cae + FR negativo + L/S ratio extremo.
Tipo C (apalancado): FR↑ + OI↑ + Taker Buy↑.



ANÁLISIS ESPECIAL: "1 de 10 que sube x20" (tu pregunta original)
Sección dedicada con:

Distribución de retornos por percentil: P10, P25, P50, P75, P90, P95, P99.
Expectativa cruda: E[R] = mean(returns).
Contribución del top 10% al PnL total: sum(top10%) / sum(all).
Frecuencia de outliers: ≥500%, ≥1000%, ≥2000%.
Kelly óptimo con distribución skewed (no asume normalidad).
Simulación Monte Carlo: tomas N señales, position size igual, ¿cuál es la distribución del PnL final?
Pregunta concreta respondida: "Si invierto $1000 igual entre 10 señales del perfil X, ¿cuál es la mediana del PnL? ¿La probabilidad de PnL>0? ¿La probabilidad de PnL>+50%? ¿La probabilidad de PnL>+200%?"


FASE 8 — Validación final

Sanity checks vs detector actual:

Comparar rsi, bb_squeeze, vol_z cuando se calculan sobre los mismos datos. Deben dar valores idénticos.
Verificar que el universo filtrado tenga sentido (BTC, ETH, SOL aparecen; tokens muertos no).


Validación contra user_history (los 13 trades cerrados):

Para cada trade, mirar qué probabilidad habría dado el modelo en entry_time.
Aunque son trades de FR arbitrage (no rallies direccionales), sirven para verificar que el modelo no está completamente desconectado de la realidad.
Limitación documentada: 13 trades es muestra muy pequeña; sirve para sanity check, no para concluir.


Test de estabilidad temporal:

Correr backtest excluyendo el último mes y ver si las predicciones del modelo coinciden con la realidad reciente.


Documentación entregable:

README.md con: instalación, uso, argumentos, ejemplos.
LIMITATIONS.md con todas las limitaciones honestas listadas.
IMPROVEMENTS.md con: cómo modificar tu sistema actual para guardar OI en funding_rate_snapshots, cómo expandir a más exchanges, cómo añadir order book data en el futuro.




Bonus: cómo arreglar el bug de OI en tu sistema actual
Tu detector actual reporta regímenes OI que son ruido inventado. Esto debería arreglarse independientemente del nuevo sistema, porque cuando empiece a guardar OI bien, en 90 días tendrás un dataset que el detector unificado puede usar.
Tarea concreta (te la documento en IMPROVEMENTS.md):

Identificar dónde tu sistema scrapea Binance/Bybit/etc. para FR.
En la misma llamada a la API, cuando es Binance, hacer también GET /fapi/v1/openInterest para obtener el OI puntual del símbolo.
Guardarlo en funding_rate_snapshots.open_interest (la columna ya existe, solo está siempre NULL).
Para otros exchanges, los endpoints son distintos pero también disponibles (Bybit: /v5/market/open-interest, OKX: /api/v5/public/open-interest).
En 30-60 días tendrás histórico de OI propio y multi-exchange — más valioso que los 30 días gratis de Binance.


Limitaciones honestas (definitivas)

Sin fees ni slippage. Para hits ≥100% no importa; para señales con expectativa <10% es la diferencia entre rentable y perdedor. Reportaremos retornos brutos con nota explícita.
Survivorship bias. Solo símbolos que existen hoy. Tokens deslistados (los más probables de pump-and-dump) no aparecen → infla rendimientos. No tiene solución sin pagar dataset histórico.
OI/L-S/Taker solo 30 días. El resto del backtest (días 0-335) tendrá esos indicadores como NaN. El modelo lo maneja pero la señal real de esos indicadores solo se valida en los últimos 30 días.
365 días incluye un solo "ciclo de mercado". Un año de cripto puede ser bull o bear o lateral; el modelo aprende ese régimen específico. Cuando cambie el régimen macro, las señales pueden degradarse.
user_history con 13 trades es sanity check, no validación estadística.
No predice quién será el ganador, identifica candidatos. De N candidatos del modelo, los datos sugieren que un porcentaje pequeño explota. Esto es exactamente el modelo "1 de 10 que sube x20" que mencionaste, y el análisis especial de Fase 7 cuantificará si es rentable o no para tu perfil.
El modelo es predictivo, no causal. Aprende correlaciones que ocurrieron en los datos. Si Binance cambia su mecanismo de funding o un protocolo grande (Ethena, etc.) altera el mercado, las correlaciones aprendidas pueden romperse.



## Criterios de éxito

- Detecta correctamente los patrones de la Fase 3 con < 5% de falsos positivos
- Escanea un proyecto de 50k LOC en menos de 10 segundos
- Output SARIF funciona con GitHub Code Scanning sin configuración adicional
- Instalable con `pip install explosive-crypto-detector`

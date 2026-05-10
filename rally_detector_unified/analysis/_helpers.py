"""
Helpers privados para los módulos de análisis.

Encapsulan los dos patrones correctos de manipulación del DataFrame de features
en formato long con DatetimeIndex no único (una fila por (timestamp, símbolo)):

- `forward_returns_by_symbol`: pct_change agrupado por símbolo, evita cruzar
  fronteras entre cripto-monedas al computar retornos forward.
- `positional_select`: selección por máscara booleana inmune a índices duplicados.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_returns_by_symbol(
    df: pd.DataFrame,
    horizon: int,
    price_col: str = "close",
    symbol_col: str = "symbol",
) -> pd.Series:
    """
    Forward return en `horizon` barras, agrupado por símbolo.

    Equivalente a `df[price_col].pct_change(horizon).shift(-horizon)` PERO
    sin cruzar fronteras entre símbolos cuando `df` tiene varias filas por
    timestamp (formato long).

    Devuelve una Series con el mismo índice y orden de filas que `df`, así
    que es compatible con máscaras booleanas derivadas del mismo `df` vía
    `positional_select`.
    """
    grouped = df.groupby(symbol_col, observed=True)[price_col]
    pct = grouped.pct_change(periods=horizon)
    # Re-group the pct series to keep .shift(-horizon) within symbol boundaries.
    return pct.groupby(df[symbol_col].values, observed=True).shift(-horizon)


def positional_select(series_or_arr, mask) -> np.ndarray:
    """
    Selección posicional inmune a índices duplicados.

    Convierte ambos argumentos a numpy arrays y aplica la máscara por posición.
    Equivale a `series.values[mask.values]` cuando ambos tienen igual longitud.
    """
    arr = np.asarray(series_or_arr)
    m = np.asarray(mask, dtype=bool)
    if len(arr) != len(m):
        raise ValueError(f"Length mismatch: arr={len(arr)} mask={len(m)}")
    return arr[m]

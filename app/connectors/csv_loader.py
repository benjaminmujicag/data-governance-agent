"""
Conector CSV — primer conector del agente de gobernanza.

Responsabilidad: leer UN archivo CSV (o TSV) y devolverlo como DataTable.

Decisiones de diseño:
- Auto-detecta el separador probando ",", ";", "\t" y "|".
- Auto-detecta el encoding probando UTF-8 primero, luego latin-1/cp1252.
  Esto es importante: los CSVs chilenos suelen venir en latin-1 con tildes.
- NO hace ninguna limpieza: devuelve los datos tal como están.
  La limpieza es responsabilidad del perfilador (Hito 2).
- Registra advertencias en metadata para que el agente las reporte.

Uso básico:
    from app.connectors.csv_loader import load_csv
    tabla = load_csv("data/publico/establecimientos_salud.csv")
    print(tabla)           # DataTable(source='csv', shape=(5707×33))
    print(tabla.df.head()) # las primeras filas
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from app.connectors.base import DataTable

logger = logging.getLogger(__name__)

# Separadores que probamos en orden de frecuencia
_CANDIDATE_SEPS = [",", ";", "\t", "|"]

# Encodings que probamos en orden de probabilidad
_CANDIDATE_ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

# Cuántas filas leemos para inferir el separador (no todo el archivo)
_SNIFF_ROWS = 20


def _detect_encoding(path: Path) -> tuple[str, bytes]:
    """
    Intenta decodificar el archivo con cada encoding candidato.
    Devuelve (encoding_que_funcionó, bytes_del_encabezado).
    Lanza ValueError si ninguno funciona.
    """
    with open(path, "rb") as f:
        raw = f.read(4096)  # los primeros 4 KB bastan para detectar

    for enc in _CANDIDATE_ENCODINGS:
        try:
            raw.decode(enc)
            return enc, raw
        except (UnicodeDecodeError, LookupError):
            continue

    raise ValueError(
        f"No se pudo detectar el encoding de {path}. "
        f"Intentado: {_CANDIDATE_ENCODINGS}"
    )


def _detect_separator(path: Path, encoding: str) -> str:
    """
    Lee las primeras _SNIFF_ROWS líneas y prueba cada separador.
    Elige el que produce más columnas de forma consistente.

    Heurística: el separador correcto es el que genera el mismo número
    de campos en TODAS las filas de muestra.
    """
    with open(path, encoding=encoding, errors="replace") as f:
        sample_lines = [f.readline() for _ in range(_SNIFF_ROWS)]

    best_sep = ","
    best_score = 0

    for sep in _CANDIDATE_SEPS:
        counts = [line.count(sep) for line in sample_lines if line.strip()]
        if not counts:
            continue
        # "Consistencia": cuántas líneas tienen exactamente la misma cantidad
        most_common = max(set(counts), key=counts.count)
        consistency = counts.count(most_common)
        col_count = most_common + 1  # nº de columnas que produciría

        # Queremos: muchas columnas Y alta consistencia
        score = col_count * consistency
        if score > best_score and col_count > 1:
            best_score = score
            best_sep = sep

    return best_sep


def load_csv(
    path: str | Path,
    sep: Optional[str] = None,
    encoding: Optional[str] = None,
    **pandas_kwargs,
) -> DataTable:
    """
    Lee un CSV y devuelve un DataTable con los datos y metadatos de carga.

    Args:
        path:           Ruta al archivo CSV.
        sep:            Separador (auto-detectado si no se especifica).
        encoding:       Encoding (auto-detectado si no se especifica).
        **pandas_kwargs: Parámetros adicionales para pd.read_csv
                         (ej.: nrows=100, skiprows=5).

    Returns:
        DataTable con .df, .source="csv", y .metadata con:
            - encoding detectado
            - separador detectado
            - shape (filas, columnas)
            - advertencias (columnas con 100% nulos, etc.)
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")

    # --- 1. Detectar encoding ---
    if encoding is None:
        encoding, _ = _detect_encoding(path)
        enc_auto = True
    else:
        enc_auto = False

    # --- 2. Detectar separador ---
    if sep is None:
        sep = _detect_separator(path, encoding)
        sep_auto = True
    else:
        sep_auto = False

    logger.info(
        "Cargando %s  |  encoding=%s%s  sep=%r%s",
        path.name,
        encoding,
        " (auto)" if enc_auto else "",
        sep,
        " (auto)" if sep_auto else "",
    )

    # --- 3. Leer el CSV ---
    warnings: list[str] = []
    try:
        df = pd.read_csv(
            path,
            sep=sep,
            encoding=encoding,
            low_memory=False,
            **pandas_kwargs,
        )
    except pd.errors.ParserError as exc:
        # Si falla, reintentamos con engine Python (más tolerante a errores)
        warnings.append(f"ParserError con engine C, reintentando con engine Python: {exc}")
        df = pd.read_csv(
            path,
            sep=sep,
            encoding=encoding,
            low_memory=False,
            engine="python",
            **pandas_kwargs,
        )

    # --- 4. Advertencias básicas ---
    total_cells = df.shape[0] * df.shape[1]
    null_pct_global = df.isnull().mean().mean() * 100

    if null_pct_global > 30:
        warnings.append(
            f"Alta tasa de nulos global: {null_pct_global:.1f}% de las celdas son NaN"
        )

    cols_all_null = df.columns[df.isnull().all()].tolist()
    if cols_all_null:
        warnings.append(f"Columnas 100% vacías: {cols_all_null}")

    for w in warnings:
        logger.warning(w)

    # --- 5. Construir metadata ---
    metadata = {
        "encoding": encoding,
        "encoding_auto_detected": enc_auto,
        "separator": sep,
        "separator_auto_detected": sep_auto,
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "total_cells": total_cells,
        "null_pct_global": round(null_pct_global, 2),
        "columns_all_null": cols_all_null,
        "warnings": warnings,
        "file_size_kb": round(path.stat().st_size / 1024, 1),
    }

    return DataTable(
        df=df,
        source="csv",
        path_or_table=str(path),
        metadata=metadata,
    )

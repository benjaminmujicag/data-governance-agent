"""
Perfilador automático de tablas — Hito 2.

Recibe un DataTable y devuelve un ProfileReport con:
  - Esquema: nombre, tipo pandas, tipo inferido.
  - Nulos reales (NaN) + nulos semánticos ("No Aplica", "", etc.).
  - Duplicados a nivel de fila.
  - Detección heurística de PII por columna.
  - Flags de calidad a nivel de columna y de tabla.

Uso:
    from app.connectors.csv_loader import load_csv
    from app.profiler.profiler import profile

    tabla = load_csv("data/publico/establecimientos_salud.csv")
    reporte = profile(tabla)
    print(reporte)
"""

import re
from pathlib import Path
from typing import Any

import pandas as pd


def _is_string_col(series: pd.Series) -> bool:
    """
    True para columnas de texto, tanto con dtype 'object' (pandas < 2)
    como con 'StringDtype' (pandas 2.x con pd.StringDtype o ArrowDtype).

    Por qué importa: pd.api.types.is_object_dtype() devuelve False para
    StringDtype, haciendo que los chequeos de texto fallen silenciosamente.
    """
    dtype_str = str(series.dtype).lower()
    # pandas 1-2: "object" | pandas 2.x StringDtype: "string[python]"
    # pandas 3.x StringDtype: "str"  ← esto cambió en pandas 3
    return dtype_str in ("object", "string", "str") or dtype_str.startswith("string")

from app.connectors.base import DataTable
from app.profiler.models import ColumnProfile, ProfileReport
from app.profiler.pii_detector import detect_pii

# ── Valores que son "nulos semánticos": parecen datos pero no lo son ──────────
_SEMANTIC_NULLS = {
    "no aplica", "no_aplica", "noapl", "n/a", "na", "null", "none",
    "sin dato", "sin información", "no disponible", "nd", "s/d",
    "desconocido", "no informado", "", " ",
}

# ── Palabras clave para inferir tipo "date" desde el nombre de columna ────────
_DATE_NAME_HINTS = re.compile(
    r"fecha|date|fec_|_fec|dt_|_dt|nacimiento|ingreso|cierre|inicio",
    re.IGNORECASE,
)

# ── Patrón para detectar fechas en strings ────────────────────────────────────
_DATE_PATTERN = re.compile(
    r"^\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}$"
)


# ── Inferencia de tipo ────────────────────────────────────────────────────────

def _infer_dtype(col_name: str, series: pd.Series) -> str:
    """
    Infiere el tipo "real" de una columna más allá de lo que dice pandas.

    pandas asigna tipos mecánicamente (si hay un string, toda la columna
    es "object"). Nosotros miramos los VALORES para entender qué es.

    Posibles resultados:
        "date"           — parece una fecha
        "numeric_int"    — entero aunque guardado como string
        "numeric_float"  — decimal aunque guardado como string
        "boolean"        — True/False, Sí/No, 1/0
        "identifier"     — código único tipo ID (alta unicidad, patrón fijo)
        "categorical"    — pocos valores únicos relativos al total
        "text"           — texto libre
        "empty"          — columna sin datos útiles
    """
    non_null = series.dropna()

    # Columna vacía
    if len(non_null) == 0:
        return "empty"

    # Ya es numérico según pandas
    if pd.api.types.is_integer_dtype(series):
        return "numeric_int"
    if pd.api.types.is_float_dtype(series):
        return "numeric_float"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    # Para cualquier tipo textual (object o StringDtype): examinamos los valores
    # Usamos astype(str) para normalizar tanto object como StringDtype a str puro.
    sample_str = non_null.astype(str).head(50)

    # ¿Es una fecha? (por nombre o por patrón de valor)
    if _DATE_NAME_HINTS.search(col_name):
        date_match = sample_str.apply(lambda v: bool(_DATE_PATTERN.match(v.strip()))).mean()
        if date_match >= 0.7:
            return "date"

    # ¿Es numérico guardado como string?
    def is_numeric(v: str) -> bool:
        try:
            float(v.replace(",", ".").strip())
            return True
        except ValueError:
            return False

    numeric_rate = sample_str.apply(is_numeric).mean()
    if numeric_rate >= 0.8:
        # ¿Entero o decimal?
        has_decimal = sample_str.apply(lambda v: "." in v or "," in v).any()
        return "numeric_float" if has_decimal else "numeric_int"

    # ¿Es booleano / Sí-No?
    bool_values = {"true", "false", "si", "no", "yes", "1", "0", "verdadero", "falso"}
    bool_rate = sample_str.str.lower().str.strip().isin(bool_values).mean()
    if bool_rate >= 0.9:
        return "boolean"

    # ¿Es categórico? (pocos valores únicos)
    n_unique = non_null.nunique()
    n_total = len(non_null)
    if n_unique <= 20 or (n_total > 0 and n_unique / n_total < 0.05):
        return "categorical"

    # ¿Es un identificador? (alta unicidad + longitud corta y uniforme)
    lengths = sample_str.str.len()
    if n_unique / n_total > 0.95 and lengths.std() < 3:
        return "identifier"

    return "text"


# ── Conteo de nulos semánticos ────────────────────────────────────────────────

def _count_semantic_nulls(series: pd.Series) -> int:
    """
    Cuenta valores que son 'nulos semánticos': no son NaN, pero tampoco
    aportan información real ("No Aplica", "N/A", "", etc.).

    Solo aplica a columnas de texto (str / object).
    """
    if not _is_string_col(series):
        return 0

    # Solo revisamos los no-nulos (los nulos reales ya los cuenta null_count)
    non_null = series.dropna().astype(str)
    return int(non_null.str.lower().str.strip().isin(_SEMANTIC_NULLS).sum())


# ── Flags de calidad por columna ──────────────────────────────────────────────

def _column_flags(
    col_name: str,
    series: pd.Series,
    dtype_inferred: str,
    null_pct: float,
    semantic_null_pct: float,
    unique_pct: float,
) -> list[str]:
    flags: list[str] = []
    n = len(series)

    if null_pct == 100.0:
        flags.append("columna_100pct_vacia")
    elif null_pct >= 80:
        flags.append(f"alta_tasa_nulos_{null_pct:.0f}pct")
    elif null_pct >= 50:
        flags.append(f"mayoria_nulos_{null_pct:.0f}pct")

    effective = null_pct + semantic_null_pct
    if semantic_null_pct > 0 and effective > null_pct + 5:
        flags.append(f"nulos_semanticos_{semantic_null_pct:.0f}pct")

    if unique_pct == 100.0 and n > 10:
        flags.append("todos_valores_unicos")  # posible ID o PII

    non_null = series.dropna()
    if non_null.nunique() == 1 and len(non_null) > 0:
        flags.append(f"un_solo_valor_unico={non_null.iloc[0]!r}")

    if dtype_inferred == "date" and _is_string_col(series):
        flags.append("fecha_guardada_como_string")

    if dtype_inferred in ("numeric_int", "numeric_float") and _is_string_col(series):
        flags.append("numerico_guardado_como_string")

    return flags


# ── Función principal ─────────────────────────────────────────────────────────

def profile(
    tabla: DataTable,
    table_name: str | None = None,
) -> ProfileReport:
    """
    Perfila una tabla y devuelve un ProfileReport.

    Args:
        tabla:       DataTable producido por cualquier conector.
        table_name:  Nombre descriptivo para el reporte.
                     Si no se da, se infiere del path/table.

    Returns:
        ProfileReport con análisis completo.
    """
    df = tabla.df
    n_rows, n_cols = df.shape

    # Nombre de tabla por defecto
    if table_name is None:
        raw = tabla.path_or_table
        table_name = Path(raw).stem if "/" in raw or "\\" in raw else raw

    # ── Duplicados a nivel de fila ────────────────────────────────────────────
    row_dup_count = int(df.duplicated().sum())
    row_dup_pct = round(row_dup_count / n_rows * 100, 2) if n_rows > 0 else 0.0

    # ── Perfilar cada columna ─────────────────────────────────────────────────
    col_profiles: list[ColumnProfile] = []

    for col in df.columns:
        series = df[col]
        non_null = series.dropna()

        # Nulos reales
        null_count = int(series.isnull().sum())
        null_pct = round(null_count / n_rows * 100, 2) if n_rows > 0 else 0.0

        # Nulos semánticos
        sem_null_count = _count_semantic_nulls(series)
        sem_null_pct = round(sem_null_count / n_rows * 100, 2) if n_rows > 0 else 0.0

        # Efectivo = real + semántico (capeado a 100)
        effective_pct = round(min(null_pct + sem_null_pct, 100.0), 2)

        # Unicidad
        unique_count = int(non_null.nunique())
        unique_pct = round(unique_count / n_rows * 100, 2) if n_rows > 0 else 0.0

        # Muestra de valores
        sample_vals: list[Any] = (
            non_null.head(5).tolist() if len(non_null) > 0 else []
        )

        # Tipo inferido
        dtype_inferred = _infer_dtype(col, series)

        # Flags de calidad
        col_quality_flags = _column_flags(
            col, series, dtype_inferred, null_pct, sem_null_pct, unique_pct
        )

        # Detección de PII
        pii_flag, pii_type = detect_pii(col, non_null)

        col_profiles.append(
            ColumnProfile(
                name=col,
                dtype_pandas=str(series.dtype),
                dtype_inferred=dtype_inferred,
                null_count=null_count,
                null_pct=null_pct,
                semantic_null_count=sem_null_count,
                semantic_null_pct=sem_null_pct,
                effective_null_pct=effective_pct,
                unique_count=unique_count,
                unique_pct=unique_pct,
                sample_values=sample_vals,
                pii_flag=pii_flag,
                pii_type=pii_type,
                quality_flags=col_quality_flags,
            )
        )

    # ── Flags a nivel de tabla ────────────────────────────────────────────────
    table_flags: list[str] = []

    # Tasa global de nulos
    global_null_pct = round(df.isnull().mean().mean() * 100, 1)
    if global_null_pct >= 30:
        table_flags.append(f"alta_tasa_nulos_global_{global_null_pct}pct")

    # Duplicados
    if row_dup_count > 0:
        table_flags.append(f"filas_duplicadas_{row_dup_count}")

    # Columnas con múltiples banderas
    multi_flag_cols = [cp.name for cp in col_profiles if len(cp.quality_flags) >= 2]
    if multi_flag_cols:
        table_flags.append(f"columnas_con_multiples_problemas={len(multi_flag_cols)}")

    # ── PII encontrada ────────────────────────────────────────────────────────
    pii_cols = [cp.name for cp in col_profiles if cp.pii_flag]

    # ── Resumen rápido ────────────────────────────────────────────────────────
    summary: dict[str, Any] = {
        "filas": n_rows,
        "columnas": n_cols,
        "duplicados_filas": row_dup_count,
        "nulos_pct_global": global_null_pct,
        "columnas_con_nulos": sum(1 for cp in col_profiles if cp.null_count > 0),
        "columnas_con_nulos_semanticos": sum(
            1 for cp in col_profiles if cp.semantic_null_count > 0
        ),
        "columnas_pii": len(pii_cols),
        "fechas_como_string": sum(
            1 for cp in col_profiles if "fecha_guardada_como_string" in cp.quality_flags
        ),
        "columnas_vacias": sum(
            1 for cp in col_profiles if "columna_100pct_vacia" in cp.quality_flags
        ),
    }

    return ProfileReport(
        table_name=table_name,
        source=tabla.source,
        path_or_table=tabla.path_or_table,
        shape=(n_rows, n_cols),
        row_duplicate_count=row_dup_count,
        row_duplicate_pct=row_dup_pct,
        columns=col_profiles,
        pii_columns=pii_cols,
        quality_flags=table_flags,
        summary=summary,
    )

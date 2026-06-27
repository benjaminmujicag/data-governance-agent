"""
Modelos de datos del perfilador — el "contrato de salida".

Por qué existen como clases separadas:
- El perfilador, el agente y el catálogo hablan de ProfileReport,
  igual que todos los conectores hablan de DataTable.
- Usar dataclasses (no dicts) da autocompletado, validación de tipos
  y un repr() legible sin código extra.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ColumnProfile:
    """
    Perfil de UNA columna de la tabla.

    Atributos:
        name               Nombre de la columna.
        dtype_pandas       Tipo que pandas asignó (str, int64, float64...).
        dtype_inferred     Tipo que el perfilador infiere que debería ser:
                           "date", "numeric_int", "numeric_float",
                           "categorical", "text", "identifier", "boolean".
        null_count         Nulos reales (NaN / None).
        null_pct           Porcentaje de nulos reales (0–100).
        semantic_null_count Valores que PARECEN datos pero son nulos
                           semánticos: "No Aplica", "N/A", "", "null", etc.
        semantic_null_pct  Porcentaje de nulos semánticos (0–100).
        effective_null_pct null_pct + semantic_null_pct (el "nulo real" útil).
        unique_count       Cantidad de valores distintos (sin contar NaN).
        unique_pct         Porcentaje de unicidad (0–100).
        sample_values      Hasta 5 valores de muestra (no nulos).
        pii_flag           True si se detectó que puede contener PII.
        pii_type           Tipo de PII detectado: "rut", "email", "telefono",
                           "nombre", "fecha_nacimiento", "salario",
                           "direccion", None.
        quality_flags      Lista de problemas detectados en esta columna.
    """

    name: str
    dtype_pandas: str
    dtype_inferred: str

    null_count: int
    null_pct: float
    semantic_null_count: int
    semantic_null_pct: float
    effective_null_pct: float

    unique_count: int
    unique_pct: float
    sample_values: list[Any]

    pii_flag: bool
    pii_type: str | None

    quality_flags: list[str] = field(default_factory=list)


@dataclass
class ProfileReport:
    """
    Reporte completo de UNA tabla.

    Atributos:
        table_name         Nombre identificador de la tabla.
        source             Origen ("csv", "postgres", etc.).
        path_or_table      Ruta o nombre de la tabla original.
        shape              (filas, columnas).
        row_duplicate_count Filas exactamente duplicadas.
        row_duplicate_pct  Porcentaje de filas duplicadas.
        columns            Lista de ColumnProfile, una por columna.
        pii_columns        Nombres de columnas con PII (subconjunto de columns).
        quality_flags      Flags a nivel de tabla (ej: "alta tasa de nulos global").
        summary            Dict con métricas rápidas para el agente.
    """

    table_name: str
    source: str
    path_or_table: str
    shape: tuple[int, int]

    row_duplicate_count: int
    row_duplicate_pct: float

    columns: list[ColumnProfile]
    pii_columns: list[str]
    quality_flags: list[str] = field(default_factory=list)

    summary: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        filas, cols = self.shape
        pii = len(self.pii_columns)
        flags = len(self.quality_flags)
        return (
            f"ProfileReport(table={self.table_name!r}, "
            f"shape=({filas}x{cols}), "
            f"pii_cols={pii}, flags={flags})"
        )

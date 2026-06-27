"""
Contrato base para todos los conectores del agente.

Un conector lee datos de UNA fuente (CSV, Postgres, BigQuery, etc.)
y devuelve siempre el mismo "paquete": un DataTable.

Por qué existe esta abstracción:
- El agente, el perfilador y el catálogo no saben ni les importa de
  dónde vienen los datos.  Solo hablan con DataTable.
- Agregar un conector nuevo = crear un módulo nuevo que devuelva
  DataTable.  El resto del sistema no cambia.
"""

from dataclasses import dataclass, field
from typing import Any
import pandas as pd


@dataclass
class DataTable:
    """
    Paquete estándar que produce cualquier conector.

    Atributos:
        df              DataFrame de pandas con los datos.
        source          Tipo de origen: "csv" | "postgres" | "bigquery" | ...
        path_or_table   Ruta al archivo o nombre de la tabla/vista.
        metadata        Diccionario libre con info técnica del origen:
                        encoding detectado, separador, cantidad de filas,
                        advertencias de carga, etc.
    """

    df: pd.DataFrame
    source: str
    path_or_table: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        rows, cols = self.df.shape
        return (
            f"DataTable(source={self.source!r}, "
            f"path_or_table={self.path_or_table!r}, "
            f"shape=({rows}x{cols}))"
        )

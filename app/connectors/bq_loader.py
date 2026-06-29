"""
Conector BigQuery — tercer conector del agente de gobernanza (Hito 9-D).

Responsabilidad: leer una tabla (o el resultado de un SELECT) desde Google BigQuery
y devolverla como DataTable, con la MISMA forma que los conectores CSV y Postgres.
Así el agente puede ingerir y perfilar datos que viven en un data warehouse, no solo
en archivos locales.

DEPENDENCIA OPCIONAL (decisión de diseño):
    La librería `google-cloud-bigquery` se importa de forma PEREZOSA (dentro de la
    función), no al inicio del módulo. Así el resto del sistema funciona aunque la
    dependencia no esté instalada; solo quien use BigQuery necesita instalarla. Si falta,
    damos un mensaje claro de instalación en vez de romper imports de todo el proyecto.

AUTENTICACIÓN (no requiere código aquí):
    El cliente de BigQuery usa Application Default Credentials (ADC). En la práctica:
      - Local:  `gcloud auth application-default login`  (o GOOGLE_APPLICATION_CREDENTIALS
                apuntando a un JSON de service account).
      - El proyecto se toma de GOOGLE_CLOUD_PROJECT/GCP_PROJECT o del argumento `project`.
    NO se hornean credenciales en el código (buena práctica de seguridad).

Uso básico (requiere credenciales GCP configuradas):
    from app.connectors.bq_loader import load_bigquery
    tabla = load_bigquery(table="mi_dataset.clientes", limit=1000)
    tabla = load_bigquery(query="SELECT rut, email FROM `proj.ds.clientes` LIMIT 100")
"""

from __future__ import annotations

import logging
import os
import re

import pandas as pd

from app.connectors.base import DataTable

logger = logging.getLogger(__name__)

# Identificador de tabla válido: project.dataset.table o dataset.table.
# Permitimos letras, dígitos, guion bajo, guion y punto (los guiones aparecen en IDs de proyecto GCP).
_TABLE_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


class BigQueryDependencyError(RuntimeError):
    """Falta la librería google-cloud-bigquery."""


def _import_bigquery():
    """Importa google-cloud-bigquery de forma perezosa, con error claro si falta."""
    try:
        from google.cloud import bigquery  # type: ignore
    except ImportError as exc:
        raise BigQueryDependencyError(
            "Para usar el conector BigQuery instala la dependencia:\n"
            "    pip install google-cloud-bigquery\n"
            "y configura credenciales GCP (gcloud auth application-default login)."
        ) from exc
    return bigquery


def load_bigquery(
    table: str | None = None,
    *,
    query: str | None = None,
    project: str | None = None,
    limit: int | None = None,
) -> DataTable:
    """
    Lee una tabla (o el resultado de un SELECT) desde BigQuery y la devuelve como DataTable.

    Debes pasar EXACTAMENTE uno de: `table` o `query`.

    Args:
        table:   Tabla en formato "dataset.tabla" o "proyecto.dataset.tabla".
        query:   Consulta SELECT en SQL estándar de BigQuery (uso de desarrollo/demo).
        project: Proyecto GCP a usar (default: el de las credenciales / GOOGLE_CLOUD_PROJECT).
        limit:   Máximo de filas (solo aplica con `table`).

    Returns:
        DataTable con .source="bigquery".

    Raises:
        ValueError:                 si no se pasa ni table ni query, o ambos, o el nombre de tabla es inválido.
        BigQueryDependencyError:    si falta la librería google-cloud-bigquery.
    """
    if (table is None) == (query is None):
        raise ValueError("Debes pasar exactamente uno de: 'table' o 'query'.")

    bigquery = _import_bigquery()

    # ── 1. Construir la consulta ──────────────────────────────────────────────
    if table is not None:
        # Validamos el identificador (no hay sql.Identifier como en psycopg; lo hacemos a mano).
        if not _TABLE_RE.match(table):
            raise ValueError(
                f"Nombre de tabla inválido: {table!r}. Use 'dataset.tabla' o 'proyecto.dataset.tabla'."
            )
        sql = f"SELECT * FROM `{table}`"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        descriptor = table
    else:
        sql = query  # type: ignore[assignment]
        descriptor = "<query>"

    project = project or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
    logger.info("Cargando desde BigQuery: %s (project=%s)", descriptor, project)

    # ── 2. Ejecutar la consulta y traer filas ─────────────────────────────────
    client = bigquery.Client(project=project)
    job = client.query(sql)
    result = job.result()  # espera a que termine el job y devuelve un RowIterator

    # Construimos el DataFrame fila por fila (cada Row se comporta como dict).
    # Evitamos result.to_dataframe() para no depender de pyarrow/db-dtypes.
    colnames = [field.name for field in result.schema]
    rows = [dict(row) for row in result]
    df = pd.DataFrame(rows, columns=colnames)

    # ── 3. Advertencias básicas (mismo criterio que CSV/Postgres) ─────────────
    warnings: list[str] = []
    n_rows, n_cols = df.shape
    if n_rows == 0:
        warnings.append("La tabla/consulta no devolvió filas.")
    null_pct_global = float(df.isnull().mean().mean() * 100) if n_cols > 0 and n_rows > 0 else 0.0
    if null_pct_global > 30:
        warnings.append(f"Alta tasa de nulos global: {null_pct_global:.1f}% de las celdas son NULL")
    for w in warnings:
        logger.warning(w)

    # ── 4. Metadata del origen ────────────────────────────────────────────────
    metadata = {
        "table": table,
        "query": query,
        "project": project,
        "limit": limit,
        "shape": {"rows": n_rows, "columns": n_cols},
        "null_pct_global": round(null_pct_global, 2),
        "bytes_processed": getattr(job, "total_bytes_processed", None),
        "warnings": warnings,
    }

    return DataTable(
        df=df,
        source="bigquery",
        path_or_table=descriptor,
        metadata=metadata,
    )

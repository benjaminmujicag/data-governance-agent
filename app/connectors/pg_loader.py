"""
Conector Postgres — segundo conector del agente de gobernanza (Hito 9-D).

Responsabilidad: leer UNA tabla (o el resultado de una consulta SELECT) desde
PostgreSQL y devolverla como DataTable, EXACTAMENTE con la misma forma que produce
el conector CSV. Así el perfilador, el agente y la API funcionan sin cambiar nada:
solo aprenden a ingerir de una fuente nueva.

Por qué esto importa para "ingesta multi-fuente":
- En una empresa los datos no viven en CSVs sueltos: viven en bases de datos.
- Demostrar que el mismo agente perfila una tabla de Postgres (no solo un archivo)
  es justo lo que pide el mercado (pipelines, conectores, gobernanza sobre la DB real).

Decisiones de diseño:
- Reutiliza el MISMO patron de conexion que app/rag/indexer.py (variables DB_* del .env).
  Hay una pequena duplicacion consciente de _conn_params (8 lineas) para que el conector
  sea independiente del modulo de RAG; si crecen los puntos de conexion, conviene extraer
  un helper compartido app/db.py.
- Compone el SQL con psycopg.sql.Identifier para EVITAR inyeccion SQL en el nombre de
  tabla/esquema (nunca interpolar identificadores con f-strings: es la puerta de entrada
  clasica al SQL injection).
- connect_timeout=5: fail-fast si la DB no responde (mismo criterio que el retriever),
  para no quedar colgados esperando una conexion que nunca llega.
- NO limpia los datos: devuelve la tabla tal cual. La limpieza/diagnostico es del perfilador.

Uso basico:
    from app.connectors.pg_loader import load_postgres
    tabla = load_postgres(table="clientes")          # lee toda la tabla "public.clientes"
    tabla = load_postgres(table="clientes", limit=100)
    tabla = load_postgres(query="SELECT rut, email FROM clientes WHERE region = 'RM'")
    print(tabla)  # DataTable(source='postgres', path_or_table='public.clientes', shape=...)
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

import pandas as pd
import psycopg
from psycopg import sql

from app.connectors.base import DataTable
from app.db import conn_params as _conn_params

logger = logging.getLogger(__name__)


class DatabaseUnavailableError(RuntimeError):
    """La base de datos no esta accesible (apagada, red, credenciales)."""


@contextmanager
def _get_conn():
    """Abre una conexion de solo lectura, la entrega y la cierra al terminar."""
    try:
        conn = psycopg.connect(**_conn_params())
    except psycopg.OperationalError as exc:
        raise DatabaseUnavailableError(
            f"No se pudo conectar a Postgres en {_conn_params()['host']}:{_conn_params()['port']}. "
            f"Verifica que el contenedor 'gobernanza-db' este levantado (docker compose up -d). "
            f"Detalle: {exc}"
        ) from exc
    try:
        yield conn
    finally:
        conn.close()


def load_postgres(
    table: str | None = None,
    *,
    query: str | None = None,
    schema: str = "public",
    limit: int | None = None,
) -> DataTable:
    """
    Lee una tabla (o el resultado de un SELECT) desde Postgres y la devuelve como DataTable.

    Debes pasar EXACTAMENTE uno de: `table` o `query`.

    Args:
        table:  Nombre de la tabla a leer (ej. "clientes"). Se compone de forma segura.
        query:  Consulta SELECT arbitraria (ej. "SELECT * FROM clientes WHERE ..."). Para
                uso de desarrollo/demo; se ejecuta tal cual.
        schema: Esquema de la tabla (default "public"). Solo aplica con `table`.
        limit:  Maximo de filas a traer (None = todas). Solo aplica con `table`.

    Returns:
        DataTable con .df, .source="postgres", .path_or_table="schema.table" (o "<query>"),
        y .metadata con shape, nulos globales y advertencias (mismo estilo que el conector CSV).

    Raises:
        ValueError:                si no se pasa ni table ni query, o si se pasan ambos.
        DatabaseUnavailableError:  si la base no responde.
    """
    if (table is None) == (query is None):
        raise ValueError("Debes pasar exactamente uno de: 'table' o 'query'.")

    # ── 1. Construir la consulta de forma segura ──────────────────────────────
    if table is not None:
        # Identifier escapa el nombre -> imposible inyectar SQL por el nombre de tabla.
        stmt = sql.SQL("SELECT * FROM {}.{}").format(
            sql.Identifier(schema), sql.Identifier(table)
        )
        if limit is not None:
            stmt = stmt + sql.SQL(" LIMIT {}").format(sql.Literal(int(limit)))
        descriptor = f"{schema}.{table}"
    else:
        stmt = sql.SQL(query)  # type: ignore[arg-type]
        descriptor = "<query>"

    logger.info("Cargando desde Postgres: %s", descriptor)

    # ── 2. Ejecutar y traer filas + nombres de columna ────────────────────────
    with _get_conn() as conn:
        cur = conn.execute(stmt)
        if cur.description is None:
            raise ValueError(
                "La consulta no devolvio filas/columnas (¿es un SELECT?). "
                "Este conector es de solo lectura."
            )
        colnames = [d.name for d in cur.description]
        rows = cur.fetchall()
        server_version = conn.info.server_version

    # pandas arma el DataFrame a partir de las filas y los nombres de columna.
    df = pd.DataFrame(rows, columns=colnames)

    # ── 3. Advertencias basicas (mismo criterio que el conector CSV) ──────────
    warnings: list[str] = []
    n_rows, n_cols = df.shape

    if n_rows == 0:
        warnings.append("La tabla/consulta no devolvio filas.")

    null_pct_global = float(df.isnull().mean().mean() * 100) if n_cols > 0 and n_rows > 0 else 0.0
    if null_pct_global > 30:
        warnings.append(f"Alta tasa de nulos global: {null_pct_global:.1f}% de las celdas son NULL")

    cols_all_null = df.columns[df.isnull().all()].tolist() if n_rows > 0 else []
    if cols_all_null:
        warnings.append(f"Columnas 100% vacias: {cols_all_null}")

    for w in warnings:
        logger.warning(w)

    # ── 4. Metadata del origen ────────────────────────────────────────────────
    metadata = {
        "schema": schema if table is not None else None,
        "table": table,
        "query": query,
        "limit": limit,
        "shape": {"rows": n_rows, "columns": n_cols},
        "null_pct_global": round(null_pct_global, 2),
        "columns_all_null": cols_all_null,
        "server_version": server_version,
        "warnings": warnings,
    }

    return DataTable(
        df=df,
        source="postgres",
        path_or_table=descriptor,
        metadata=metadata,
    )

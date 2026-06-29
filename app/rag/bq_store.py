"""
bq_store.py — Vector store del RAG sobre BigQuery (backend alternativo a pgVector).

POR QUÉ EXISTE
  El RAG necesita guardar embeddings y recuperar los más parecidos a una pregunta.
  pgVector (app/rag/indexer.py + retriever.py) hace eso en Postgres. Este módulo hace
  LO MISMO en BigQuery, para poder desplegar en Cloud Run usando solo GCP (sin una base
  externa). Se elige el backend con la variable de entorno RAG_BACKEND=bigquery.

DECISIÓN DE DISEÑO (importante para entender el código):
  Para un corpus pequeño (≈23 fragmentos) NO usamos un índice vectorial (VECTOR_SEARCH +
  CREATE VECTOR INDEX). Un índice IVF está pensado para escala (miles/millones de vectores)
  y da resultados APROXIMADOS para ganar velocidad. Con 23 filas, comparar contra todas
  (búsqueda exacta / brute force) es instantáneo y exacto. Usamos la función nativa
  COSINE_DISTANCE(v1, v2), que da la distancia coseno (0 = idénticos, 2 = opuestos);
  similitud = 1 - distancia. Para escala futura: migrar a VECTOR_SEARCH con índice.

AUTENTICACIÓN
  bigquery.Client() usa Application Default Credentials (ADC):
    - Local:      gcloud auth application-default login
    - Cloud Run:  la identidad (service account) del servicio, vía IAM. No hay contraseña
                  de DB que guardar como secreto: una ventaja sobre un Postgres gestionado.

ESQUEMA de la tabla (mismo concepto que governance_chunks en pgVector):
  doc_name STRING | chunk_id INT64 | text STRING | char_start INT64 | char_end INT64 |
  embedding ARRAY<FLOAT64>  (REPEATED FLOAT64)
"""

from __future__ import annotations

import os

from app.rag.chunker import Chunk

# Nombre de la tabla (mismo que en pgVector, para mantener el paralelismo conceptual).
TABLE_NAME = "governance_chunks"


class BigQueryDependencyError(RuntimeError):
    """Falta la librería google-cloud-bigquery."""


def _import_bigquery():
    """Importa google-cloud-bigquery de forma perezosa, con error claro si falta."""
    try:
        from google.cloud import bigquery  # type: ignore
    except ImportError as exc:
        raise BigQueryDependencyError(
            "Para usar el backend BigQuery instala la dependencia:\n"
            "    pip install google-cloud-bigquery\n"
            "y configura credenciales GCP (gcloud auth application-default login)."
        ) from exc
    return bigquery


def _project() -> str | None:
    """Proyecto GCP a usar (default: el de las credenciales / GOOGLE_CLOUD_PROJECT)."""
    return os.getenv("BQ_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")


def _dataset() -> str:
    """Dataset de BigQuery donde vive la tabla (default 'gobernanza')."""
    return os.getenv("BQ_DATASET", "gobernanza")


def _location() -> str:
    """Región del dataset (default 'US', elegible para free tier)."""
    return os.getenv("BQ_LOCATION", "US")


def _client():
    """Crea un cliente de BigQuery autenticado por ADC."""
    bigquery = _import_bigquery()
    return bigquery.Client(project=_project(), location=_location())


def _table_ref(client) -> str:
    """Devuelve el identificador completo `proyecto.dataset.tabla`."""
    return f"{client.project}.{_dataset()}.{TABLE_NAME}"


def create_table(drop_first: bool = False) -> None:
    """
    Crea el dataset (si no existe) y la tabla de chunks en BigQuery.

    Args:
        drop_first: Si True, borra y recrea la tabla. En BigQuery la carga idempotente
                    se hace con WRITE_TRUNCATE (ver insert_chunks), así que normalmente
                    no hace falta drop_first.
    """
    bigquery = _import_bigquery()
    client = _client()

    # 1) Dataset (idempotente).
    dataset_id = f"{client.project}.{_dataset()}"
    dataset = bigquery.Dataset(dataset_id)
    dataset.location = _location()
    client.create_dataset(dataset, exists_ok=True)

    # 2) Tabla.
    table_id = _table_ref(client)
    if drop_first:
        client.delete_table(table_id, not_found_ok=True)

    schema = [
        bigquery.SchemaField("doc_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("chunk_id", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("text", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("char_start", "INT64"),
        bigquery.SchemaField("char_end", "INT64"),
        # ARRAY<FLOAT64> se declara como FLOAT64 en modo REPEATED.
        bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
    ]
    table = bigquery.Table(table_id, schema=schema)
    client.create_table(table, exists_ok=True)


def insert_chunks(chunks: list[Chunk], embeddings: list[list[float]]) -> int:
    """
    Carga chunks + embeddings en BigQuery, reemplazando el contenido (WRITE_TRUNCATE).

    Por qué WRITE_TRUNCATE en vez de INSERT con ON CONFLICT (como en pgVector):
      BigQuery es un data warehouse analítico, no transaccional; no tiene upsert por clave
      simple. Como el corpus es FIJO y se reindexa de una vez, lo más limpio y idempotente
      es truncar y recargar: re-correr deja exactamente el mismo estado, sin duplicados.

    Returns:
        Número de filas cargadas.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks ({len(chunks)}) y embeddings ({len(embeddings)}) deben tener la misma longitud."
        )

    bigquery = _import_bigquery()
    client = _client()
    create_table(drop_first=False)  # asegura que exista
    table_id = _table_ref(client)

    rows = [
        {
            "doc_name": c.doc_name,
            "chunk_id": c.chunk_id,
            "text": c.text,
            "char_start": c.char_start,
            "char_end": c.char_end,
            "embedding": list(emb),
        }
        for c, emb in zip(chunks, embeddings)
    ]

    schema = [
        bigquery.SchemaField("doc_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("chunk_id", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("text", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("char_start", "INT64"),
        bigquery.SchemaField("char_end", "INT64"),
        bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
    ]
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    job = client.load_table_from_json(rows, table_id, job_config=job_config)
    job.result()  # espera a que termine la carga
    return len(rows)


def count_chunks() -> int:
    """Devuelve el total de filas (chunks) en la tabla."""
    client = _client()
    table_id = _table_ref(client)
    row = next(iter(client.query(f"SELECT COUNT(*) AS n FROM `{table_id}`").result()))
    return int(row["n"])


def list_documents() -> list[str]:
    """Devuelve los documentos únicos indexados."""
    client = _client()
    table_id = _table_ref(client)
    rows = client.query(
        f"SELECT DISTINCT doc_name FROM `{table_id}` ORDER BY doc_name"
    ).result()
    return [row["doc_name"] for row in rows]


def search(query_embedding: list[float], k: int = 5, min_score: float = 0.3) -> list[dict]:
    """
    Recupera los k chunks más cercanos al embedding de la query, por distancia coseno.

    Devuelve dicts {doc_name, chunk_id, text, score} con score = similitud coseno [0,1]
    (1 - distancia). Filtra por min_score. retriever.py los envuelve en RetrievedChunk.
    """
    bigquery = _import_bigquery()
    client = _client()
    table_id = _table_ref(client)

    # @q = embedding de la query (parámetro tipado, evita inyección y problemas de formato).
    sql = f"""
        SELECT
            doc_name,
            chunk_id,
            text,
            1 - COSINE_DISTANCE(embedding, @q) AS similarity
        FROM `{table_id}`
        ORDER BY COSINE_DISTANCE(embedding, @q)
        LIMIT @k
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("q", "FLOAT64", query_embedding),
            bigquery.ScalarQueryParameter("k", "INT64", k),
        ]
    )
    rows = client.query(sql, job_config=job_config).result()

    results = []
    for row in rows:
        score = round(float(row["similarity"]), 4)
        if score >= min_score:
            results.append(
                {
                    "doc_name": row["doc_name"],
                    "chunk_id": int(row["chunk_id"]),
                    "text": row["text"],
                    "score": score,
                }
            )
    return results

"""
indexer.py — Crea la tabla de chunks en pgVector e inserta embeddings.

Por qué pgVector:
  Es una extensión de Postgres (ya levantada en Docker). No hay que aprender una nueva DB.
  Soporta búsqueda kNN con similitud coseno usando el operador <=>.
  Para corpus de ~200-500 chunks, pgVector es más que suficiente.

Qué hace este módulo:
  1. Crea (o valida) la tabla governance_chunks con columnas: id, doc_name, chunk_id,
     text, embedding (vector de dim 3072).
  2. Inserta chunks + embeddings de forma IDEMPOTENTE: si el mismo (doc_name, chunk_id)
     ya existe, no lo duplica (ON CONFLICT DO NOTHING).
  3. Crea un índice IVFFlat para acelerar la búsqueda cuando el corpus crezca.

Qué es IVFFlat:
  Índice aproximado de vectores. Divide el espacio vectorial en "celdas" y al buscar
  solo examina las celdas más cercanas. Mucho más rápido que recorrer todos los vectores.
  lists=100 es razonable para corpus < 10.000 chunks.

SDK de DB: psycopg3 (psycopg[binary]) — la versión 3 del driver Python de Postgres.
"""

from __future__ import annotations
import os
from contextlib import contextmanager

import psycopg
from dotenv import load_dotenv

from app.rag.chunker import Chunk

load_dotenv()

VECTOR_DIM = 3072
TABLE_NAME = "governance_chunks"


def _conn_params() -> dict:
    """Lee los parámetros de conexión del .env (mismos que docker-compose.yml)."""
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "dbname": os.getenv("DB_NAME", "gobernanza"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "postgres"),
    }


@contextmanager
def _get_conn():
    """Context manager que abre una conexión, hace commit y la cierra."""
    conn = psycopg.connect(**_conn_params())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def create_table(drop_first: bool = False) -> None:
    """
    Crea la tabla governance_chunks en pgVector si no existe.

    Args:
        drop_first: Si True, borra y recrea la tabla (útil para re-indexar desde cero).
                    En uso normal deja False (idempotente).
    """
    with _get_conn() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        if drop_first:
            conn.execute(f"DROP TABLE IF EXISTS {TABLE_NAME};")

        # La columna embedding es de tipo vector(3072) — tipo que provee pgvector
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id         SERIAL PRIMARY KEY,
                doc_name   TEXT NOT NULL,
                chunk_id   INTEGER NOT NULL,
                text       TEXT NOT NULL,
                char_start INTEGER,
                char_end   INTEGER,
                embedding  vector({VECTOR_DIM}),
                UNIQUE (doc_name, chunk_id)
            );
        """)

        # NOTA sobre índices:
        # IVFFlat tiene un límite de 2000 dimensiones y nuestros vectores son de 3072.
        # Para corpus pequeños (< 10.000 chunks), la búsqueda secuencial es suficientemente
        # rápida (milisegundos). Cuando el corpus crezca, usar HNSW con pgvector >= 0.7:
        #   CREATE INDEX USING hnsw (embedding vector_cosine_ops);
        # HNSW no tiene límite de dimensiones y es más rápido que IVFFlat.


def insert_chunks(chunks: list[Chunk], embeddings: list[list[float]]) -> int:
    """
    Inserta chunks y sus embeddings en pgVector de forma idempotente.

    Si (doc_name, chunk_id) ya existe, lo omite sin lanzar error.

    Args:
        chunks:     Lista de Chunk del chunker.
        embeddings: Lista de vectores, mismo orden que chunks.

    Returns:
        Número de chunks efectivamente insertados.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks ({len(chunks)}) y embeddings ({len(embeddings)}) deben tener la misma longitud."
        )

    inserted_total = 0

    with _get_conn() as conn:
        for chunk, emb in zip(chunks, embeddings):
            # pgvector acepta el vector como string '[f1,f2,...,fn]' con cast ::vector
            vector_str = "[" + ",".join(str(x) for x in emb) + "]"

            cur = conn.execute(
                f"""
                INSERT INTO {TABLE_NAME}
                    (doc_name, chunk_id, text, char_start, char_end, embedding)
                VALUES (%s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT (doc_name, chunk_id) DO NOTHING
                """,
                (
                    chunk.doc_name,
                    chunk.chunk_id,
                    chunk.text,
                    chunk.char_start,
                    chunk.char_end,
                    vector_str,
                ),
            )
            inserted_total += cur.rowcount

    return inserted_total


def count_chunks() -> int:
    """Devuelve el total de chunks indexados en la tabla."""
    with _get_conn() as conn:
        row = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};").fetchone()
        return row[0]


def list_documents() -> list[str]:
    """Devuelve la lista de documentos únicos en el índice."""
    with _get_conn() as conn:
        rows = conn.execute(
            f"SELECT DISTINCT doc_name FROM {TABLE_NAME} ORDER BY doc_name;"
        ).fetchall()
        return [row[0] for row in rows]

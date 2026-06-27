"""
retriever.py — Recupera los chunks más relevantes para una pregunta.

Cómo funciona (el "R" de RAG):
  1. Convierte la pregunta en un vector (embed_query).
  2. Busca en pgVector los k chunks con menor distancia coseno al vector de la pregunta.
  3. La distancia coseno con <=> devuelve valores entre 0 y 2:
       0   = vectores idénticos (máxima relevancia)
       1   = vectores ortogonales (sin relación)
       2   = vectores opuestos
  4. Transformamos: similitud = 1 - distancia, así 1.0 = más relevante.
  5. Filtramos por min_score para descartar resultados de baja relevancia.

Por qué top-k y no todos los chunks:
  El LLM tiene una ventana de contexto limitada. Con top-5 tenemos suficiente contexto
  para la mayoría de preguntas sin introducir "ruido" que confunda al LLM.

SDK de DB: psycopg3.
"""

from __future__ import annotations
import os
from dataclasses import dataclass

import psycopg
from dotenv import load_dotenv

from app.rag.embedder import embed_query

load_dotenv()

TABLE_NAME = "governance_chunks"


@dataclass
class RetrievedChunk:
    """Fragmento recuperado del índice, listo para incluir en el prompt del LLM."""
    doc_name: str     # nombre del documento fuente (es la "cita")
    chunk_id: int     # posición del chunk dentro del documento
    text: str         # texto del fragmento
    score: float      # similitud coseno [0, 1] — más alto = más relevante


def _conn_params() -> dict:
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "dbname": os.getenv("DB_NAME", "gobernanza"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "postgres"),
    }


def retrieve(query: str, k: int = 5, min_score: float = 0.3) -> list[RetrievedChunk]:
    """
    Recupera los k fragmentos más relevantes para la query dada.

    Proceso:
      1. Embeddea la query con task_type="RETRIEVAL_QUERY" (diferente al de indexación).
      2. Hace búsqueda kNN en pgVector con <=> (distancia coseno).
      3. Filtra resultados con similitud >= min_score.

    Args:
        query:     Pregunta del usuario en lenguaje natural.
        k:         Número máximo de fragmentos a recuperar.
        min_score: Umbral mínimo de similitud (0.0–1.0). 0.3 es conservador razonable.

    Returns:
        Lista de RetrievedChunk ordenada de mayor a menor similitud.
    """
    query_vector = embed_query(query)
    vector_str = "[" + ",".join(str(x) for x in query_vector) + "]"

    conn = psycopg.connect(**_conn_params())
    try:
        # <=> = distancia coseno; transformamos a similitud = 1 - distancia
        rows = conn.execute(
            f"""
            SELECT
                doc_name,
                chunk_id,
                text,
                1 - (embedding <=> %s::vector) AS similarity
            FROM {TABLE_NAME}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """,
            (vector_str, vector_str, k),
        ).fetchall()
    finally:
        conn.close()

    results = [
        RetrievedChunk(
            doc_name=row[0],
            chunk_id=row[1],
            text=row[2],
            score=round(float(row[3]), 4),
        )
        for row in rows
        if float(row[3]) >= min_score
    ]

    return results


def format_context_for_llm(chunks: list[RetrievedChunk]) -> str:
    """
    Formatea los chunks recuperados como contexto para el prompt del LLM.

    El formato incluye la cita explícita de la fuente, lo que permite al LLM
    responder anclado en los documentos reales e incluir citas verificables.

    Args:
        chunks: Lista de chunks recuperados por retrieve().

    Returns:
        Bloque de texto listo para insertar en el prompt del agente.
    """
    if not chunks:
        return "No se encontraron fragmentos relevantes en el corpus de gobernanza."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[FUENTE {i}] Documento: {chunk.doc_name} | Fragmento #{chunk.chunk_id} "
            f"| Relevancia: {chunk.score:.2f}\n"
            f"{chunk.text}"
        )

    return "\n\n---\n\n".join(parts)

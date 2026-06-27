"""
embedder.py — Convierte texto en vectores de embedding usando Gemini.

Qué es un embedding:
  Imagina un mapa donde cada texto tiene coordenadas que representan su significado.
  "datos sensibles" y "información personal confidencial" quedan cerca en ese mapa,
  aunque no compartan palabras. El embedding es ese vector de números (3072 dimensiones).

Por qué Gemini gemini-embedding-001:
  - Dimensión 3072 (alta calidad semántica), confirmada en Hito 0.
  - Free tier de Google AI Studio → costo $0 durante el desarrollo.

IMPORTANTE — task_type:
  Gemini optimiza el embedding según el uso:
  - RETRIEVAL_DOCUMENT: para indexar documentos del corpus.
  - RETRIEVAL_QUERY: para embeddear la pregunta del usuario.
  Usar el tipo correcto mejora la calidad de la recuperación.

SDK usado: google-genai (nuevo), patrón: genai.Client(api_key=...).models.embed_content(...)
"""

from __future__ import annotations
import os
import time

from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv

load_dotenv()

_REQUEST_DELAY_SECONDS = 0.1  # respetar free tier (1500 req/min)


def _make_client() -> genai.Client:
    """Crea un cliente Gemini usando la GOOGLE_API_KEY del .env."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.startswith("tu_api_key"):
        raise EnvironmentError(
            "GOOGLE_API_KEY no encontrada o no configurada. "
            "Asegúrate de tener el .env con la clave de Google AI Studio."
        )
    return genai.Client(api_key=api_key)


def _get_model_name() -> str:
    """Lee el modelo de embedding del .env (default: gemini-embedding-001)."""
    return os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")


def embed_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """
    Embeddea un texto y devuelve el vector (lista de floats de dim 3072).

    Args:
        text:       Texto a embeddear (chunk o pregunta del usuario).
        task_type:  "RETRIEVAL_DOCUMENT" para corpus, "RETRIEVAL_QUERY" para preguntas.

    Returns:
        Lista de 3072 floats representando el significado del texto.
    """
    client = _make_client()
    model = _get_model_name()

    result = client.models.embed_content(
        model=model,
        contents=text,
        config=genai_types.EmbedContentConfig(task_type=task_type),
    )
    time.sleep(_REQUEST_DELAY_SECONDS)
    return list(result.embeddings[0].values)


def embed_batch(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    batch_size: int = 20,
) -> list[list[float]]:
    """
    Embeddea una lista de textos, mostrando progreso cada batch_size elementos.

    Por qué no en paralelo:
      El free tier de Gemini tiene límite de requests por minuto. Hacemos un delay
      de 0.1s entre requests para no superar el límite y evitar errores 429.

    Args:
        texts:      Lista de textos a embeddear.
        task_type:  Task type aplicado a todos los textos.
        batch_size: Cada cuántos textos imprimir progreso.

    Returns:
        Lista de vectores, uno por texto, en el mismo orden.
    """
    embeddings: list[list[float]] = []

    for i, text in enumerate(texts):
        embedding = embed_text(text, task_type=task_type)
        embeddings.append(embedding)

        if (i + 1) % batch_size == 0 or (i + 1) == len(texts):
            print(f"  Embeddings: {i + 1}/{len(texts)}")

    return embeddings


def embed_query(query: str) -> list[float]:
    """
    Shortcut para embeddear una pregunta del usuario (RETRIEVAL_QUERY).

    Siempre usa esta función para queries (no embed_text) para obtener
    el task_type correcto y mayor precisión de recuperación.
    """
    return embed_text(query, task_type="RETRIEVAL_QUERY")

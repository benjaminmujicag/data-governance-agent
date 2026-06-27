"""
chunker.py — Divide documentos de texto en chunks con solapamiento.

Por qué hacemos chunking:
  - Los documentos de política tienen 400-800 palabras. Un embedding de 3072 dim
    representa un trozo de texto; si el trozo es demasiado largo, el vector promedia
    demasiado significado y pierde precisión al recuperar.
  - Con chunks de ~300 palabras y un overlap de ~50, si una pregunta toca el borde
    de un chunk, el chunk vecino también lo captura.

Trade-off tamaño/overlap:
  - Chunks muy pequeños (< 100 palabras): alta precisión pero poca coherencia.
  - Chunks muy grandes (> 500 palabras): mejor contexto pero menor precisión de recuperación.
  - 300 palabras / 50 de overlap: buen equilibrio para documentos de política.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    """Un fragmento de documento listo para embeddear e indexar."""
    doc_name: str          # nombre del archivo fuente (ej.: "politica_pii_y_datos_personales")
    chunk_id: int          # índice 0-based del chunk dentro del documento
    text: str              # texto del chunk (lo que se embeddea y se almacena)
    char_start: int        # posición de inicio en el documento original (para trazabilidad)
    char_end: int          # posición de fin en el documento original


def chunk_text(
    text: str,
    doc_name: str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[Chunk]:
    """
    Parte 'text' en chunks de 'chunk_size' palabras con 'overlap' palabras de solapamiento.

    Ejemplo con chunk_size=5, overlap=2 y texto "A B C D E F G H":
      chunk 0: A B C D E
      chunk 1: D E F G H   ← empieza en D, que es la posición 5-2=3

    Args:
        text:       Texto completo del documento.
        doc_name:   Nombre identificador del documento (sin extensión).
        chunk_size: Número de palabras por chunk.
        overlap:    Palabras que se repiten entre chunks consecutivos.

    Returns:
        Lista de Chunk, en orden de aparición en el documento.
    """
    words = text.split()
    if not words:
        return []

    step = max(1, chunk_size - overlap)
    chunks: list[Chunk] = []
    chunk_id = 0

    i = 0
    while i < len(words):
        window = words[i : i + chunk_size]
        chunk_text_str = " ".join(window)

        # Calcular posición de caracteres en el texto original (para trazabilidad)
        char_start = len(" ".join(words[:i])) + (1 if i > 0 else 0)
        char_end = char_start + len(chunk_text_str)

        chunks.append(
            Chunk(
                doc_name=doc_name,
                chunk_id=chunk_id,
                text=chunk_text_str,
                char_start=char_start,
                char_end=char_end,
            )
        )
        chunk_id += 1
        i += step

    return chunks


def chunk_file(
    path: Path | str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[Chunk]:
    """
    Lee un archivo .txt y devuelve sus chunks.

    Args:
        path:       Ruta al archivo de texto.
        chunk_size: Número de palabras por chunk.
        overlap:    Palabras de solapamiento entre chunks consecutivos.

    Returns:
        Lista de Chunk.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    doc_name = path.stem  # nombre sin extensión, ej.: "politica_pii_y_datos_personales"
    return chunk_text(text, doc_name=doc_name, chunk_size=chunk_size, overlap=overlap)


def chunk_directory(
    directory: Path | str,
    pattern: str = "*.txt",
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[Chunk]:
    """
    Chunkea todos los archivos .txt de un directorio.

    Args:
        directory:  Directorio con los documentos del corpus.
        pattern:    Glob de archivos a procesar.
        chunk_size: Número de palabras por chunk.
        overlap:    Palabras de solapamiento.

    Returns:
        Lista de todos los chunks de todos los documentos, en orden de archivo.
    """
    directory = Path(directory)
    all_chunks: list[Chunk] = []
    for filepath in sorted(directory.glob(pattern)):
        file_chunks = chunk_file(filepath, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(file_chunks)
    return all_chunks

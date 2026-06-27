"""
indexar_corpus.py — Construye el índice RAG en pgVector a partir del corpus de gobernanza.

Qué hace este script (en orden):
  1. Lee todos los .txt de data/politicas/
  2. Los divide en chunks de 300 palabras con 50 de overlap
  3. Genera el embedding de cada chunk con Gemini (gemini-embedding-001, dim=3072)
  4. Crea la tabla governance_chunks en pgVector (si no existe)
  5. Inserta los chunks + embeddings (idempotente: no duplica si ya existen)

Cómo verificar que salió bien:
  - El script imprime cuántos chunks total y cuántos insertó vs. cuántos ya existían.
  - Corre `scripts/test_rag.py` después para verificar que la búsqueda funciona.

Precondiciones:
  - Docker Desktop ABIERTO (docker compose up -d en la carpeta del proyecto).
  - .env con GOOGLE_API_KEY válida.
  - venv activado: .venv\\Scripts\\activate (Windows) o source .venv/bin/activate (Mac/Linux).

Costo estimado: ~$0.00 (Gemini embedding free tier cubre esto con holgura).
"""

import sys
from pathlib import Path

# Agregar la raíz del proyecto al path para importar app.*
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.chunker import chunk_directory, Chunk
from app.rag.embedder import embed_batch
from app.rag.indexer import create_table, insert_chunks, count_chunks, list_documents

POLITICAS_DIR = PROJECT_ROOT / "data" / "politicas"
CHUNK_SIZE = 300
OVERLAP = 50


def main():
    print("=" * 60)
    print("INDEXADOR DE CORPUS DE GOBERNANZA")
    print("=" * 60)

    # 1. Chunking
    print(f"\n[1/4] Dividiendo documentos en chunks...")
    print(f"      Directorio: {POLITICAS_DIR}")
    print(f"      Configuración: {CHUNK_SIZE} palabras por chunk, {OVERLAP} de overlap")

    chunks: list[Chunk] = chunk_directory(
        POLITICAS_DIR,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
    )

    if not chunks:
        print("ERROR: No se encontraron archivos .txt en data/politicas/")
        print("Verifica que el directorio existe y tiene los archivos de políticas.")
        sys.exit(1)

    # Resumen por documento
    docs_seen = {}
    for c in chunks:
        docs_seen[c.doc_name] = docs_seen.get(c.doc_name, 0) + 1

    print(f"      Total chunks generados: {len(chunks)}")
    for doc, n in sorted(docs_seen.items()):
        print(f"        - {doc}: {n} chunks")

    # 2. Embeddings
    print(f"\n[2/4] Generando embeddings con Gemini...")
    print(f"      Esto puede tomar ~{len(chunks) * 0.1:.0f}–{len(chunks) * 0.2:.0f} segundos (free tier delay).")
    print(f"      Modelo: models/gemini-embedding-exp-03-07 (dim=3072)")

    texts = [c.text for c in chunks]
    embeddings = embed_batch(texts, task_type="RETRIEVAL_DOCUMENT")

    print(f"      Embeddings generados: {len(embeddings)}")
    print(f"      Dimensión del vector: {len(embeddings[0])}")

    # 3. Crear tabla en pgVector
    print(f"\n[3/4] Preparando tabla en pgVector...")
    create_table(drop_first=False)  # idempotente
    print(f"      Tabla '{TABLE_NAME_INFO}' lista.")

    # 4. Insertar
    print(f"\n[4/4] Insertando en pgVector (idempotente)...")
    inserted = insert_chunks(chunks, embeddings)
    total_in_db = count_chunks()
    docs_in_db = list_documents()

    print(f"      Chunks nuevos insertados: {inserted}")
    print(f"      Total chunks en el índice: {total_in_db}")
    print(f"      Documentos indexados ({len(docs_in_db)}):")
    for doc in docs_in_db:
        print(f"        - {doc}")

    print("\n" + "=" * 60)
    print("INDEXACIÓN COMPLETADA")
    print(f"Próximo paso: python scripts/test_rag.py")
    print("=" * 60)


# Usamos esta variable solo para el mensaje informativo (el nombre real está en indexer.py)
TABLE_NAME_INFO = "governance_chunks"

if __name__ == "__main__":
    main()

"""
test_rag.py — Verifica que el RAG funciona correctamente.

Qué verifica:
  - Para cada pregunta de prueba, recupera los top-5 chunks más relevantes.
  - Verifica que el documento esperado aparece en los resultados (al menos 1 de los 5).
  - Imprime los resultados para que puedas ver qué fragmentos se recuperaron.

Cómo interpretar los resultados:
  - score cercano a 1.0 = alta relevancia
  - score ~ 0.5 = relevancia moderada
  - score < 0.3 = probablemente ruido (filtrado por min_score)

Precondiciones:
  - scripts/indexar_corpus.py debe haberse corrido exitosamente primero.
  - Docker Desktop abierto y docker compose up -d activo.

Cómo correr:
  cd proyectos/asistente-rag-gobernanza
  .venv\\Scripts\\activate   (Windows)
  python scripts/test_rag.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.retriever import retrieve, format_context_for_llm
from app.rag.indexer import count_chunks, list_documents

# Pares (pregunta, documento_esperado)
# El documento_esperado es el nombre del .txt sin extensión
TEST_CASES = [
    (
        "¿Qué tipos de datos se consideran sensibles según la política de privacidad?",
        "politica_pii_y_datos_personales",
    ),
    (
        "¿Cuánto tiempo debo guardar los datos de transacciones financieras de clientes?",
        "politica_retencion_datos",
    ),
    (
        "¿Cómo se clasifican los datos según su nivel de confidencialidad?",
        "politica_clasificacion_datos",
    ),
    (
        "¿Qué dice la ley chilena sobre notificación de brechas de seguridad?",
        "resumen_ley_21719",
    ),
    (
        "¿Puedo usar el RUT de un cliente para análisis de marketing?",
        "politica_pii_y_datos_personales",
    ),
]


def main():
    print("=" * 70)
    print("TEST DEL MÓDULO RAG — Recuperación de fragmentos de gobernanza")
    print("=" * 70)

    # Verificar que hay datos en el índice
    total = count_chunks()
    docs = list_documents()
    if total == 0:
        print("\nERROR: El índice está vacío.")
        print("Corre primero: python scripts/indexar_corpus.py")
        sys.exit(1)

    print(f"\nÍndice: {total} chunks | {len(docs)} documentos")
    print(f"Documentos: {', '.join(docs)}\n")

    passed = 0
    failed = 0

    for i, (query, expected_doc) in enumerate(TEST_CASES, 1):
        print(f"-" * 70)
        print(f"TEST {i}: {query}")
        print(f"Doc esperado: {expected_doc}")
        print()

        results = retrieve(query, k=5, min_score=0.3)

        if not results:
            print("  RESULTADO: Sin resultados (todos por debajo de min_score=0.3)")
            print(f"  [FALLO] no se recuperaron fragmentos relevantes")
            failed += 1
            continue

        # Verificar si el doc esperado aparece en los resultados
        found_docs = [r.doc_name for r in results]
        hit = expected_doc in found_docs

        for j, r in enumerate(results, 1):
            marker = " <--" if r.doc_name == expected_doc else ""
            print(f"  [{j}] {r.doc_name} | chunk #{r.chunk_id} | score={r.score:.3f}{marker}")
            # Mostrar preview del texto (primeras 120 chars)
            preview = r.text[:120].replace("\n", " ")
            print(f"      \"{preview}...\"")

        if hit:
            print(f"\n  [OK] documento esperado encontrado (posicion {found_docs.index(expected_doc) + 1})")
            passed += 1
        else:
            print(f"\n  [FALLO] '{expected_doc}' no aparecio en los top-{len(results)}")
            failed += 1

        print()

    print("=" * 70)
    print(f"RESULTADO FINAL: {passed}/{len(TEST_CASES)} tests pasaron")
    if failed == 0:
        print("Todos los tests pasaron. El RAG está funcionando correctamente.")
        print("Próximo paso: Hito 4 — Agente con tools (function calling).")
    else:
        print(f"\n{failed} test(s) fallaron. Posibles causas:")
        print("  1. El corpus no fue indexado (corre indexar_corpus.py primero).")
        print("  2. Los embeddings del corpus y la query no son comparables")
        print("     (verificar que se usó RETRIEVAL_DOCUMENT al indexar).")
        print("  3. min_score muy alto — prueba bajar a 0.2 en retrieve().")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()

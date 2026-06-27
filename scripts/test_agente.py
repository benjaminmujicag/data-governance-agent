"""
test_agente.py — Prueba el agente de extremo a extremo (Hito 4).

Qué verifica:
  Para cada una de las 5 preguntas del CONTEXTO, corre el agente completo y comprueba:
    1. Que el agente usó al menos una tool (no respondió "de memoria").
    2. Que usó la tool ESPERADA para ese tipo de pregunta (calidad/PII -> profile_table,
       políticas/ley -> search_governance).
    3. Imprime la respuesta y la traza de tools para inspección manual.

Cómo interpretar:
  - [OK]    = el agente usó la(s) tool(s) correcta(s) y respondió.
  - [FALLO] = no usó la tool esperada (puede haber alucinado o elegido mal).
  Lee SIEMPRE las respuestas: que use la tool correcta no garantiza que la respuesta
  sea perfecta. La evaluación rigurosa de calidad llega en el Hito 7 (LLM-as-Judge).

Precondiciones:
  - Docker Desktop ABIERTO y `docker compose up -d` (el corpus vive en pgVector).
  - Corpus indexado: `python scripts/indexar_corpus.py`.
  - `.env` con GOOGLE_API_KEY válida.

Cómo correr:
  cd proyectos/asistente-rag-gobernanza
  .venv\\Scripts\\activate   (Windows)
  python scripts/test_agente.py
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.agent import ask
from app.agent import catalog

# El free tier de Gemini permite ~10 generate_content por minuto. Cada pregunta hace
# ~2 llamadas (pedir tool + redactar), así que 5 preguntas = ~10 llamadas. Pausamos entre
# preguntas para repartirlas y no agotar la cuota de golpe. El reintento con backoff del
# agente es la red de seguridad si aun así se cruza el límite.
SLEEP_BETWEEN_QUESTIONS_S = 8

# (pregunta, tool_esperada). tool_esperada = la herramienta que DEBERÍA usar.
TEST_CASES = [
    (
        "¿Qué problemas de calidad tiene la tabla de establecimientos de salud?",
        "profile_table",
    ),
    (
        "¿Qué columnas tienen PII en la tabla de clientes?",
        "profile_table",
    ),
    (
        "¿Cuánto tiempo dice la política de retención que debo guardar los datos de clientes?",
        "search_governance",
    ),
    (
        "¿Qué dice la Ley 21.719 sobre datos sensibles?",
        "search_governance",
    ),
    (
        "¿Puedo usar el RUT de un cliente para análisis de marketing?",
        "search_governance",
    ),
]


def main():
    print("=" * 70)
    print("TEST DEL AGENTE — Function calling de extremo a extremo (Hito 4)")
    print("=" * 70)

    passed = 0
    failed = 0

    for i, (question, expected_tool) in enumerate(TEST_CASES, 1):
        # Pausa entre preguntas (excepto antes de la primera) para respetar el rate limit.
        if i > 1:
            time.sleep(SLEEP_BETWEEN_QUESTIONS_S)

        # Limpiamos la pizarra entre preguntas para que cada test sea independiente.
        catalog.reset()

        print("-" * 70)
        print(f"PREGUNTA {i}: {question}")
        print(f"Tool esperada: {expected_tool}")
        print()

        result = ask(question, verbose=True)

        tools_usadas = [tc.name for tc in result.trace]
        uso_tool = len(tools_usadas) > 0
        uso_tool_esperada = expected_tool in tools_usadas

        print()
        print(f"Tools usadas ({result.steps} pasos): {tools_usadas or 'ninguna'}")
        print("\nRESPUESTA DEL AGENTE:")
        print(result.answer)
        print()

        if uso_tool and uso_tool_esperada:
            print(f"  [OK] usó la tool esperada '{expected_tool}'")
            passed += 1
        elif not uso_tool:
            print("  [FALLO] el agente NO usó ninguna tool (posible alucinación)")
            failed += 1
        else:
            print(f"  [FALLO] no usó '{expected_tool}'. Usó: {tools_usadas}")
            failed += 1

        print()

    print("=" * 70)
    print(f"RESULTADO FINAL: {passed}/{len(TEST_CASES)} tests pasaron")
    if failed == 0:
        print("Todos los tests pasaron. El agente usa las tools correctamente.")
        print("Próximo paso: Hito 5 — API FastAPI.")
    else:
        print(f"\n{failed} test(s) fallaron. Posibles causas:")
        print("  1. Docker apagado o corpus sin indexar (search_governance falla).")
        print("  2. La descripción de la tool no es clara y el modelo eligió mal.")
        print("  3. El modelo respondió de memoria sin llamar tools (revisar SYSTEM_PROMPT).")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()

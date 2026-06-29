"""
evaluar.py — Corre la evaluacion del agente contra el golden set y escribe un reporte.

QUE HACE (en orden):
  1. Carga data/evaluacion/golden_set.json.
  2. Para cada caso: corre el agente y lo juzga con el LLM-as-Judge (ver app/evaluation/judge.py).
  3. Calcula metricas: respuesta_correcta_%, cita_correcta_%, tool_correcta_%, score_promedio.
  4. Escribe el reporte en reportes/evaluacion_agente.md (métricas reproducibles del agente).

POR QUE UN REPORTE EN MARKDOWN
  Las metricas convierten una demo en ingenieria medible. Guardarlo en reportes/ DENTRO del
  proyecto lo deja versionado en git y listo para citar en el README.

CUIDADO CON LA CUOTA (free tier de Gemini)
  Cada caso hace ~2-3 llamadas del agente + 1 del juez. El golden set completo (24 casos)
  son ~80-100 llamadas y puede agotar la cuota diaria. RECOMENDACION: corre primero un
  subconjunto con --limit 3 para validar el flujo, y luego el set completo cuando tengas cupo.

PRECONDICIONES:
  - Docker Desktop ABIERTO y `docker compose up -d` (el corpus vive en pgVector).
  - Corpus indexado: `python scripts/indexar_corpus.py`.
  - .env con GOOGLE_API_KEY y, idealmente, LLM_MODEL=gemini-2.5-flash.
  - venv activado.

COMO CORRER:
  cd proyectos/asistente-rag-gobernanza
  .venv\\Scripts\\activate            (Windows)
  python scripts/evaluar.py --limit 3        # prueba rapida
  python scripts/evaluar.py                  # set completo
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.judge import evaluate_golden_set, EvalSummary, CaseResult

GOLDEN_SET_PATH = PROJECT_ROOT / "data" / "evaluacion" / "golden_set.json"
REPORTE_PATH = PROJECT_ROOT / "reportes" / "evaluacion_agente.md"


def cargar_golden_set(path: Path, limit: int | None) -> list[dict]:
    """Lee el golden set y devuelve la lista de casos (opcionalmente acotada a 'limit')."""
    data = json.loads(path.read_text(encoding="utf-8"))
    casos = data["casos"]
    if limit is not None:
        casos = casos[:limit]
    return casos


def _emoji_bool(b: bool) -> str:
    """Marca visual para el reporte/consola sin depender de emojis (Windows cp1252)."""
    return "OK" if b else "X"


def construir_reporte_md(summary: EvalSummary, modelo_info: dict) -> str:
    """Arma el contenido Markdown del reporte de evaluacion."""
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")
    lineas: list[str] = []

    lineas.append("# Reporte de evaluacion del agente de gobernanza")
    lineas.append("")
    lineas.append(f"- **Fecha:** {fecha}")
    lineas.append(f"- **Casos evaluados:** {summary.n_ok}/{summary.n} "
                  f"(corridos sin error / total)")
    lineas.append(f"- **Modelo del agente:** `{modelo_info.get('agente', '?')}`")
    lineas.append(f"- **Modelo del juez:** `{modelo_info.get('juez', '?')}`")
    lineas.append("")

    lineas.append("## Metricas globales")
    lineas.append("")
    lineas.append("| Metrica | Valor |")
    lineas.append("|---|---|")
    lineas.append(f"| Respuesta correcta | {summary.respuesta_correcta_pct:.1f}% |")
    lineas.append(f"| Cita correcta | {summary.cita_correcta_pct:.1f}% |")
    lineas.append(f"| Tool correcta (deterministico) | {summary.tool_correcta_pct:.1f}% |")
    lineas.append(f"| Score promedio (1-5) | {summary.score_promedio:.2f} |")
    lineas.append("")

    # Metricas por categoria (utiles para ver donde falla el agente).
    lineas.append("## Por categoria")
    lineas.append("")
    lineas.append("| Categoria | N | Correctas | Citas OK | Score prom |")
    lineas.append("|---|---|---|---|---|")
    categorias = sorted({r.categoria for r in summary.resultados})
    for cat in categorias:
        grupo = [r for r in summary.resultados if r.categoria == cat and r.error is None]
        if not grupo:
            lineas.append(f"| {cat} | 0 | - | - | - |")
            continue
        n = len(grupo)
        corr = 100.0 * sum(1 for r in grupo if r.correcta) / n
        cita = 100.0 * sum(1 for r in grupo if r.cita_correcta) / n
        score = sum(r.score for r in grupo) / n
        lineas.append(f"| {cat} | {n} | {corr:.0f}% | {cita:.0f}% | {score:.2f} |")
    lineas.append("")

    lineas.append("## Detalle por caso")
    lineas.append("")
    for r in summary.resultados:
        lineas.append(f"### {r.id} — {r.categoria}")
        lineas.append(f"**Pregunta:** {r.pregunta}")
        lineas.append("")
        if r.error:
            lineas.append(f"> ERROR al evaluar este caso: {r.error}")
            lineas.append("")
            continue
        lineas.append(f"- Correcta: **{_emoji_bool(r.correcta)}** | "
                      f"Cita: **{_emoji_bool(r.cita_correcta)}** | "
                      f"Score: **{r.score}/5**")
        lineas.append(f"- Fuente esperada: `{r.fuente_esperada}` | "
                      f"Tool esperada: `{r.tool_esperada}` | "
                      f"Tools usadas: `{r.tools_usadas or 'ninguna'}` "
                      f"({_emoji_bool(r.tool_correcta)})")
        lineas.append(f"- Juez: {r.justificacion}")
        lineas.append("")
        lineas.append("<details><summary>Respuesta del agente</summary>")
        lineas.append("")
        lineas.append("```")
        lineas.append(r.respuesta_agente.strip())
        lineas.append("```")
        lineas.append("</details>")
        lineas.append("")

    return "\n".join(lineas)


def imprimir_resumen_consola(summary: EvalSummary) -> None:
    """Imprime el resumen en la terminal (para no tener que abrir el .md cada vez)."""
    print("=" * 70)
    print("RESUMEN DE LA EVALUACION")
    print("=" * 70)
    print(f"Casos sin error : {summary.n_ok}/{summary.n}")
    print(f"Respuesta correcta : {summary.respuesta_correcta_pct:.1f}%")
    print(f"Cita correcta      : {summary.cita_correcta_pct:.1f}%")
    print(f"Tool correcta      : {summary.tool_correcta_pct:.1f}%")
    print(f"Score promedio     : {summary.score_promedio:.2f}/5")
    print("-" * 70)
    for r in summary.resultados:
        if r.error:
            print(f"  {r.id} [{r.categoria}] ERROR: {r.error}")
        else:
            print(
                f"  {r.id} [{r.categoria}] correcta={_emoji_bool(r.correcta)} "
                f"cita={_emoji_bool(r.cita_correcta)} score={r.score} "
                f"tool={_emoji_bool(r.tool_correcta)}"
            )
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluacion del agente con golden set + LLM-as-Judge.")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Evalua solo los primeros N casos (util para no gastar cuota). Ej: --limit 3",
    )
    parser.add_argument(
        "--sleep", type=float, default=8.0,
        help="Segundos de pausa entre casos (anti rate-limit del free tier). Default 8.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="No imprime el progreso caso a caso.",
    )
    args = parser.parse_args()

    if not GOLDEN_SET_PATH.exists():
        print(f"ERROR: no se encontro el golden set en {GOLDEN_SET_PATH}")
        sys.exit(1)

    casos = cargar_golden_set(GOLDEN_SET_PATH, args.limit)
    print(f"Cargados {len(casos)} casos del golden set"
          + (f" (limitado a {args.limit})" if args.limit else "") + ".")
    print("Corriendo agente + juez por cada caso (esto consume cuota de Gemini)...\n")

    summary = evaluate_golden_set(casos, sleep_s=args.sleep, verbose=not args.quiet)

    # Info de modelos para el encabezado del reporte (se leen igual que en el codigo).
    import os
    modelo_info = {
        "agente": os.getenv("LLM_MODEL", "gemini-2.5-flash-lite"),
        "juez": os.getenv("JUDGE_MODEL", os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")),
    }

    REPORTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORTE_PATH.write_text(construir_reporte_md(summary, modelo_info), encoding="utf-8")

    print()
    imprimir_resumen_consola(summary)
    print(f"\nReporte escrito en: {REPORTE_PATH}")

    # Salimos con codigo != 0 si hubo casos con error (util para CI mas adelante).
    hubo_errores = any(r.error for r in summary.resultados)
    sys.exit(1 if hubo_errores else 0)


if __name__ == "__main__":
    main()

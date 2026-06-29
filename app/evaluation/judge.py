"""
judge.py — Evaluacion del agente con un golden set y un LLM-as-Judge.

QUE PROBLEMA RESUELVE
  Sabemos que el agente "responde", pero no QUE TAN BIEN. Para medirlo necesitamos:
    1. Un golden set: preguntas con su respuesta correcta y su fuente esperada (la "pauta").
    2. Una forma de comparar la respuesta del agente con la pauta. Como las respuestas son
       texto libre ("guardalo 10 anos" == "el plazo es de diez anos"), no sirve comparar
       strings con ==. Usamos OTRO LLM como JUEZ que evalua si la respuesta es correcta.

POR QUE UN LLM JUZGA A OTRO (LLM-as-Judge)
  Un juez humano no escala (24 preguntas x cada cambio de codigo = horas). Un LLM puede
  leer la pregunta, la respuesta esperada y la respuesta del agente, y emitir un veredicto
  en segundos. Analogia: un profesor corrigiendo CONTRA UNA PAUTA, no inventando el criterio.

SESGOS DEL JUEZ (a tener presentes)
  - self-preference: un modelo tiende a premiar respuestas de su propia familia.
  - verbosity bias: tiende a premiar respuestas largas aunque no sean mejores.
  Mitigaciones aplicadas aqui:
    a) Le DAMOS la respuesta esperada como ancla (no inventa el criterio de verdad).
    b) JUDGE_MODEL es configurable: idealmente un modelo distinto al del agente.
    c) Pedimos salida ESTRUCTURADA (JSON con esquema) para no parsear texto a mano.

QUE ES Y QUE NO ES TRABAJO DEL LLM AQUI
  - El LLM juzga lo SEMANTICO: ¿la respuesta es correcta? ¿cita la fuente correcta?
  - La verificacion de la TOOL usada es DETERMINISTA (la leemos de la traza del agente):
    no necesita un LLM, es gratis y 100% confiable. Buen recordatorio: no todo se resuelve
    con un modelo; cuando tienes la verdad exacta, un chequeo en Python es mejor.

SDK: google-genai. Reutilizamos _make_client y _generate_with_retry del agente para
heredar el manejo de cuota (429 por minuto/dia, 503 transitorio) ya probado en el Hito 4.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from google.genai import types
from pydantic import BaseModel, Field

from app.agent.agent import ask, _make_client, _generate_with_retry, _get_model_name
from app.agent import catalog
import os


# ── Esquema de salida del juez (salida estructurada) ────────────────────────────
# Le pasamos este esquema al modelo via response_schema. El SDK fuerza al modelo a
# responder un JSON con EXACTAMENTE estos campos, y nos lo devuelve ya parseado.
class JudgeVerdict(BaseModel):
    """Veredicto del juez sobre una respuesta del agente."""
    correcta: bool = Field(
        description="True si la respuesta del agente es factualmente correcta y coherente "
        "con la respuesta esperada (la pauta). Errores de hecho o contradicciones => False."
    )
    cita_correcta: bool = Field(
        description="True si la respuesta cita/menciona la fuente esperada. Si la fuente "
        "esperada es 'ninguna', es True solo cuando el agente declara que no tiene base y "
        "NO inventa una fuente."
    )
    score: int = Field(
        description="Calidad global de 1 (muy mala) a 5 (excelente), considerando exactitud, "
        "completitud y si cito la fuente."
    )
    justificacion: str = Field(
        description="Una o dos frases en espanol explicando el veredicto."
    )


JUDGE_SYSTEM_PROMPT = """\
Eres un evaluador EXPERTO y ESTRICTO de un asistente de gobernanza de datos.
Tu trabajo NO es responder la pregunta, sino JUZGAR la respuesta de un agente contra una
respuesta esperada (la pauta de correccion) que te entregamos como verdad de referencia.

Criterios:
1. correcta: la respuesta del agente debe coincidir en lo SUSTANCIAL con la respuesta
   esperada. Puede usar otras palabras; lo que importa son los HECHOS (plazos, cifras,
   nombres, prohibiciones). Si contradice la pauta o inventa hechos, es incorrecta.
2. cita_correcta: la respuesta debe apoyarse en la fuente esperada. La fuente puede
   aparecer como nombre de documento o de tabla, o describirse claramente (p. ej.
   'la politica de retencion'). Si la fuente esperada es 'ninguna', cita_correcta es True
   SOLO si el agente reconoce que no tiene base suficiente y NO inventa una fuente.
3. score: 5 = correcta y bien citada; 3 = parcialmente correcta o cita debil;
   1 = incorrecta, vacia o alucinada.

Se justo pero exigente. No premies respuestas largas que no contienen los hechos correctos.
Responde SOLO con el objeto estructurado solicitado.
"""


def _get_judge_model() -> str:
    """
    Modelo del juez. Por defecto reutiliza el del agente, pero idealmente se configura
    un modelo DISTINTO (JUDGE_MODEL en .env) para reducir el self-preference bias.
    """
    return os.getenv("JUDGE_MODEL", _get_model_name())


def _build_judge_input(
    pregunta: str,
    respuesta_esperada: str,
    fuente_esperada: str,
    respuesta_agente: str,
) -> str:
    """Arma el texto que ve el juez: pregunta + pauta + fuente + respuesta a evaluar."""
    return (
        f"PREGUNTA:\n{pregunta}\n\n"
        f"RESPUESTA ESPERADA (pauta de correccion):\n{respuesta_esperada}\n\n"
        f"FUENTE ESPERADA (documento o tabla que deberia citarse): {fuente_esperada}\n\n"
        f"RESPUESTA DEL AGENTE (a evaluar):\n{respuesta_agente}\n\n"
        "Emite tu veredicto."
    )


def judge_answer(
    pregunta: str,
    respuesta_esperada: str,
    fuente_esperada: str,
    respuesta_agente: str,
    *,
    verbose: bool = False,
) -> JudgeVerdict:
    """
    Pide al LLM-juez un veredicto sobre UNA respuesta del agente.

    Returns:
        JudgeVerdict con correcta/cita_correcta/score/justificacion.
    """
    client = _make_client()
    model = _get_judge_model()

    config = types.GenerateContentConfig(
        system_instruction=JUDGE_SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=JudgeVerdict,
        temperature=0,  # juez determinista: mismo input -> mismo veredicto
    )
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(
                text=_build_judge_input(
                    pregunta, respuesta_esperada, fuente_esperada, respuesta_agente
                )
            )],
        )
    ]

    response = _generate_with_retry(client, model, contents, config, verbose=verbose)

    # El SDK ya intenta instanciar el pydantic en response.parsed. Si por alguna razon
    # no viene, hacemos fallback a parsear el texto JSON manualmente.
    verdict = getattr(response, "parsed", None)
    if isinstance(verdict, JudgeVerdict):
        return _clamp_score(verdict)

    data = json.loads(response.text)
    return _clamp_score(JudgeVerdict(**data))


def _clamp_score(v: JudgeVerdict) -> JudgeVerdict:
    """Asegura que score quede en [1, 5] aunque el modelo se salga del rango."""
    v.score = max(1, min(5, int(v.score)))
    return v


# ── Resultados de la evaluacion ─────────────────────────────────────────────────
@dataclass
class CaseResult:
    """Resultado de evaluar UN caso del golden set."""
    id: str
    categoria: str
    pregunta: str
    fuente_esperada: str
    tool_esperada: str
    respuesta_agente: str
    tools_usadas: list[str]
    tool_correcta: bool       # determinista: ¿uso la tool esperada? (de la traza)
    correcta: bool            # del juez
    cita_correcta: bool       # del juez
    score: int                # del juez (1-5)
    justificacion: str
    error: str | None = None  # si algo fallo al correr este caso


@dataclass
class EvalSummary:
    """Resumen agregado de toda la corrida de evaluacion."""
    resultados: list[CaseResult] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.resultados)

    @property
    def n_ok(self) -> int:
        """Casos que corrieron sin error (el agente respondio y el juez emitio veredicto)."""
        return sum(1 for r in self.resultados if r.error is None)

    def _pct(self, cond) -> float:
        evaluables = [r for r in self.resultados if r.error is None]
        if not evaluables:
            return 0.0
        return 100.0 * sum(1 for r in evaluables if cond(r)) / len(evaluables)

    @property
    def respuesta_correcta_pct(self) -> float:
        return self._pct(lambda r: r.correcta)

    @property
    def cita_correcta_pct(self) -> float:
        return self._pct(lambda r: r.cita_correcta)

    @property
    def tool_correcta_pct(self) -> float:
        return self._pct(lambda r: r.tool_correcta)

    @property
    def score_promedio(self) -> float:
        evaluables = [r for r in self.resultados if r.error is None]
        if not evaluables:
            return 0.0
        return sum(r.score for r in evaluables) / len(evaluables)


def evaluate_golden_set(
    casos: list[dict],
    *,
    sleep_s: float = 8.0,
    verbose: bool = False,
) -> EvalSummary:
    """
    Corre el agente + el juez sobre cada caso del golden set y agrega metricas.

    Por cada caso:
      1. catalog.reset() para que cada caso sea independiente (no arrastrar estado).
      2. ask(pregunta) -> respuesta del agente + traza de tools.
      3. tool_correcta = (tool_esperada en la traza)  [determinista]
      4. judge_answer(...) -> veredicto del LLM-juez  [semantico]
      5. Pausa sleep_s entre casos para respetar el rate limit del free tier.

    Nota de cuota: cada caso hace ~2-3 llamadas del agente + 1 del juez. Con 24 casos
    son ~80-100 llamadas. Si el free tier se agota, usa un golden set acotado (--limit).

    Args:
        casos:   Lista de dicts del golden set (cada uno con pregunta/respuesta_esperada/...).
        sleep_s: Segundos de pausa entre casos (anti rate-limit).
        verbose: Si True, imprime el progreso de cada caso.

    Returns:
        EvalSummary con la lista de CaseResult y las metricas agregadas.
    """
    summary = EvalSummary()

    for i, caso in enumerate(casos, 1):
        cid = caso.get("id", f"caso_{i}")
        pregunta = caso["pregunta"]
        respuesta_esperada = caso["respuesta_esperada"]
        fuente_esperada = caso.get("fuente_esperada", "")
        tool_esperada = caso.get("tool_esperada", "")
        categoria = caso.get("categoria", "")

        if i > 1:
            time.sleep(sleep_s)

        if verbose:
            print(f"[{i}/{len(casos)}] {cid}: {pregunta}")

        catalog.reset()

        try:
            result = ask(pregunta, verbose=False)
            tools_usadas = [tc.name for tc in result.trace]
            tool_correcta = tool_esperada in tools_usadas

            verdict = judge_answer(
                pregunta=pregunta,
                respuesta_esperada=respuesta_esperada,
                fuente_esperada=fuente_esperada,
                respuesta_agente=result.answer,
                verbose=verbose,
            )

            summary.resultados.append(
                CaseResult(
                    id=cid,
                    categoria=categoria,
                    pregunta=pregunta,
                    fuente_esperada=fuente_esperada,
                    tool_esperada=tool_esperada,
                    respuesta_agente=result.answer,
                    tools_usadas=tools_usadas,
                    tool_correcta=tool_correcta,
                    correcta=verdict.correcta,
                    cita_correcta=verdict.cita_correcta,
                    score=verdict.score,
                    justificacion=verdict.justificacion,
                )
            )

            if verbose:
                print(
                    f"      -> correcta={verdict.correcta} cita={verdict.cita_correcta} "
                    f"score={verdict.score} tool_ok={tool_correcta}"
                )

        except Exception as exc:  # noqa: BLE001
            # Un caso que falla (cuota agotada, DB caida, etc.) no debe tumbar toda la
            # evaluacion: lo registramos como error y seguimos con los demas.
            if verbose:
                print(f"      -> ERROR: {exc}")
            summary.resultados.append(
                CaseResult(
                    id=cid,
                    categoria=categoria,
                    pregunta=pregunta,
                    fuente_esperada=fuente_esperada,
                    tool_esperada=tool_esperada,
                    respuesta_agente="",
                    tools_usadas=[],
                    tool_correcta=False,
                    correcta=False,
                    cita_correcta=False,
                    score=1,
                    justificacion="",
                    error=str(exc),
                )
            )

    return summary

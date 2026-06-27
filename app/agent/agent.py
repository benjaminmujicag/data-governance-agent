"""
agent.py — El agente: LLM (Gemini) + tools + loop ReAct manual.

Qué es el loop ReAct (Reason + Act):
  El agente alterna entre PENSAR (el modelo decide qué hacer) y ACTUAR (ejecutamos
  la tool que pidió). El ciclo:

    1. Le mandamos al modelo la conversación + las tools disponibles.
    2. El modelo responde una de dos cosas:
         a) Uno o más "function_call" → quiere usar herramientas.
         b) Texto final → ya tiene la respuesta.
    3. Si pidió tools: las ejecutamos, le devolvemos los resultados, y volvemos al paso 1.
    4. Si dio texto: terminamos y devolvemos esa respuesta.

  El límite max_steps evita bucles infinitos (un agente mal calibrado podría pedir
  tools para siempre). Es una red de seguridad obligatoria en sistemas con LLMs.

Por qué el loop MANUAL (y no el automático del SDK):
  El SDK puede ejecutar las tools por ti si le pasas funciones Python. Lo hacemos a mano
  para VER el mecanismo: cómo el modelo pide una tool, cómo le devolvemos el resultado con
  role="tool", y cómo decide seguir o parar. Ver esto es lo que permite debuggear un agente.

SDK: google-genai (nuevo). Patrón confirmado:
  - types.FunctionDeclaration / types.Tool         → declarar tools
  - client.models.generate_content(tools=[...])    → llamar al modelo con tools
  - response.function_calls                        → lista de llamadas que pidió el modelo
  - types.Part.from_function_response(...)         → empaquetar el resultado de la tool
  - types.Content(role="tool", parts=[...])        → devolvérselo al modelo
"""

from __future__ import annotations
import os
import time
from dataclasses import dataclass, field

from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from dotenv import load_dotenv

from app.agent.tools import TOOL, DISPATCH

load_dotenv()

# Códigos de error TRANSITORIOS de la API que vale la pena reintentar:
#   429 = cuota agotada (free tier: 10 req/min). Espera a que libere la ventana de 1 min.
#   500 / 503 = el servidor de Google está saturado. Reintentar suele resolverlo.
# Un 4xx que NO sea 429 (400, 401, 403) NO se reintenta: es culpa nuestra (request o key mal).
_RETRYABLE_CODES = {429, 500, 503}


SYSTEM_PROMPT = """\
Eres un asistente de GOBERNANZA DE DATOS para una empresa de banca/retail en Chile.

Tu trabajo es responder preguntas sobre:
- CALIDAD de los datos de una tabla (nulos, duplicados, tipos inconsistentes).
- PII (datos personales sensibles) presente en las tablas.
- POLÍTICAS internas de datos y la LEY 21.719 (Protección de Datos, Chile).

Reglas obligatorias:
1. NO inventes. Si la respuesta depende de una política, una ley o los datos de una tabla,
   USA las herramientas para obtener la información antes de responder.
2. Para preguntas de calidad o PII de una tabla → usa profile_table.
3. Para preguntas de políticas, normas, plazos o ley → usa search_governance.
4. Si una herramienta no devuelve información útil (por ejemplo, search_governance
   responde con n_resultados=0, o una tabla no existe), NO inventes una respuesta:
   declara explícitamente que no encontraste base suficiente y sugiere qué fuente
   consultar. Es preferible decir "no tengo base para responder" a alucinar.
5. Responde SIEMPRE en español, claro y conciso.
6. Termina SIEMPRE con una sección '[CITAS]' que liste las fuentes que usaste:
   el nombre del documento (para políticas/ley) o el nombre de la tabla (para calidad/PII).
   Si no usaste ninguna fuente, escribe '[CITAS] Ninguna'.
"""


def _make_client() -> genai.Client:
    """Crea el cliente Gemini con la GOOGLE_API_KEY del .env."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.startswith("tu_api_key"):
        raise EnvironmentError(
            "GOOGLE_API_KEY no encontrada o no configurada. Revisa el archivo .env."
        )
    return genai.Client(api_key=api_key)


def _get_model_name() -> str:
    """Modelo de generación del .env (default: gemini-2.5-flash-lite)."""
    return os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")


def _generate_with_retry(
    client: genai.Client,
    model: str,
    contents: list[types.Content],
    config: types.GenerateContentConfig,
    max_retries: int = 5,
    verbose: bool = False,
):
    """
    Llama a generate_content con reintentos ante errores TRANSITORIOS (429/500/503).

    Usa backoff exponencial (5, 10, 20, 40s, cap 60s) para dar tiempo a que el servidor
    se recupere (503) o a que la ventana de 1 minuto libere cupo del free tier (429).
    Los errores NO transitorios (400/401/403) se propagan de inmediato: reintentar no
    los arreglaría porque la culpa es del request/credenciales, no del servidor.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return client.models.generate_content(
                model=model, contents=contents, config=config
            )
        except genai_errors.APIError as exc:
            code = getattr(exc, "code", None)

            # No todos los 429 son iguales: el de cuota DIARIA (PerDay) no se arregla
            # esperando segundos (solo se resetea a medianoche). Fallamos rápido con un
            # mensaje claro en vez de malgastar el backoff.
            if code == 429 and "PerDay" in str(exc):
                raise RuntimeError(
                    "Cuota DIARIA del free tier agotada para el modelo "
                    f"'{model}'. Opciones: esperar al reset (medianoche hora del "
                    "Pacífico), cambiar LLM_MODEL a otro modelo con cuota disponible, "
                    "o habilitar facturación. (Reintentar no ayuda: es límite por día.)"
                ) from exc

            if code not in _RETRYABLE_CODES or attempt == max_retries:
                raise
            wait = min(60, 5 * 2 ** (attempt - 1))
            if verbose:
                print(f"  [reintento {attempt}/{max_retries}] API {code} transitorio; espero {wait}s...")
            time.sleep(wait)


@dataclass
class ToolCall:
    """Registro de una tool que el agente decidió ejecutar (para trazabilidad)."""
    name: str
    args: dict
    ok: bool


@dataclass
class AgentResult:
    """Resultado de una corrida del agente."""
    answer: str
    trace: list[ToolCall] = field(default_factory=list)
    steps: int = 0


def ask(question: str, max_steps: int = 6, verbose: bool = False) -> AgentResult:
    """
    Ejecuta el loop ReAct para responder una pregunta.

    Args:
        question:  Pregunta del usuario en lenguaje natural.
        max_steps: Máximo de vueltas del loop (red de seguridad anti-bucle).
        verbose:   Si True, imprime cada decisión del agente (didáctico/debug).

    Returns:
        AgentResult con la respuesta final, la traza de tools usadas y el nº de pasos.
    """
    client = _make_client()
    model = _get_model_name()

    config = types.GenerateContentConfig(
        tools=[TOOL],
        system_instruction=SYSTEM_PROMPT,
    )

    # 'contents' es la conversación completa que crece en cada vuelta del loop.
    contents: list[types.Content] = [
        types.Content(role="user", parts=[types.Part.from_text(text=question)])
    ]

    trace: list[ToolCall] = []

    for step in range(1, max_steps + 1):
        response = _generate_with_retry(
            client, model, contents, config, verbose=verbose
        )

        candidate = response.candidates[0]
        # Guardamos el turno del modelo en la conversación (incluye sus function_call).
        contents.append(candidate.content)

        function_calls = response.function_calls  # None o lista de FunctionCall

        # Caso (b): no pidió tools → es la respuesta final.
        if not function_calls:
            if verbose:
                print(f"[paso {step}] respuesta final del modelo")
            return AgentResult(answer=response.text or "", trace=trace, steps=step)

        # Caso (a): pidió una o más tools → las ejecutamos.
        tool_response_parts: list[types.Part] = []
        for fc in function_calls:
            name = fc.name
            args = dict(fc.args) if fc.args else {}

            if verbose:
                print(f"[paso {step}] el agente llama -> {name}({args})")

            func = DISPATCH.get(name)
            if func is None:
                # El modelo pidió una tool inexistente: devolvemos error, no crasheamos.
                result_payload = {"error": f"Tool desconocida: {name}"}
                ok = False
            else:
                try:
                    result_payload = {"result": func(**args)}
                    ok = True
                except Exception as exc:  # noqa: BLE001
                    # Le devolvemos el error al modelo para que lo maneje (p. ej. reintente).
                    result_payload = {"error": str(exc)}
                    ok = False

            trace.append(ToolCall(name=name, args=args, ok=ok))
            tool_response_parts.append(
                types.Part.from_function_response(name=name, response=result_payload)
            )

        # Devolvemos TODOS los resultados de tools al modelo (role="tool") y seguimos.
        contents.append(types.Content(role="tool", parts=tool_response_parts))

    # Si llegamos aquí, el agente nunca dio respuesta final dentro de max_steps.
    return AgentResult(
        answer=(
            "No pude llegar a una respuesta final dentro del límite de pasos. "
            "Esto suele indicar que el agente entró en bucle pidiendo herramientas."
        ),
        trace=trace,
        steps=max_steps,
    )

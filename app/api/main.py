"""
main.py — La API HTTP del agente de gobernanza (FastAPI).

Esta capa NO tiene lógica de negocio: solo EXPONE por HTTP lo que ya construimos
(las tools del Hito 4 y el agente). Es el "mesero": recibe el pedido, llama a la
cocina (tools/agente) y devuelve el plato. Por eso es delgada.

Endpoints:
  GET  /health   → ¿está viva la API? (no toca LLM ni DB)
  GET  /catalog  → lista tablas cargadas y disponibles (no toca LLM)
  POST /connect  → carga un CSV y lo registra en el catálogo
  POST /profile  → perfila una tabla (esquema, nulos, PII, calidad)
  POST /ask      → el agente responde una pregunta usando sus tools

Cómo levantar el servidor (uvicorn = el proceso que escucha en el puerto):
  cd proyectos/asistente-rag-gobernanza
  .venv\\Scripts\\activate
  uvicorn app.api.main:app --reload
  Luego abre http://localhost:8000/docs  (documentación interactiva autogenerada por FastAPI).

Manejo de errores:
  Traducimos las excepciones a códigos HTTP correctos en vez de devolver un 500 genérico:
    - archivo no encontrado          → 404
    - cuota diaria del LLM agotada   → 503 (servicio no disponible temporalmente)
  Esto es parte de hacer una API "de verdad": el cliente debe saber QUÉ salió mal.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

from app.api.schemas import (
    ConnectRequest,
    ConnectPgRequest,
    ProfileRequest,
    AskRequest,
    AskResponse,
    HealthResponse,
)
from app.api.ratelimit import rate_limit_ask
from app.agent import tools
from app.agent.agent import ask as agent_ask
from app.connectors.pg_loader import DatabaseUnavailableError

app = FastAPI(
    title="Agente de Gobernanza de Datos",
    description="API que expone un agente RAG con tools para gobernanza de datos.",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Chequeo de vida. Útil para monitoreo y para verificar que el server levantó."""
    return HealthResponse(status="ok", service="agente-gobernanza")


@app.get("/catalog")
def get_catalog() -> dict:
    """Lista tablas cargadas/perfiladas en la sesión y las disponibles para perfilar."""
    return tools.list_catalog()


@app.post("/connect")
def connect(req: ConnectRequest) -> dict:
    """Carga un CSV desde una ruta y lo registra en el catálogo en memoria."""
    try:
        return tools.connect_csv(req.path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/connect-postgres")
def connect_postgres(req: ConnectPgRequest) -> dict:
    """Carga una tabla desde Postgres y la registra en el catálogo en memoria."""
    try:
        return tools.connect_postgres(req.table, schema=req.schema_name, limit=req.limit)
    except DatabaseUnavailableError as exc:
        # La base no responde (apagada / red / credenciales) → 503 servicio no disponible.
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/profile")
def profile(req: ProfileRequest) -> dict:
    """Perfila una tabla y devuelve su digest (esquema, nulos, PII, flags de calidad)."""
    result = tools.profile_table(req.table_name)
    # profile_table devuelve {"error": ...} si no reconoce la tabla → lo traducimos a 404.
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=404, detail=result)
    return result


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, _rl: None = Depends(rate_limit_ask)) -> AskResponse:
    """
    El agente responde la pregunta usando function calling sobre sus tools.

    Depends(rate_limit_ask) se ejecuta ANTES del cuerpo: si se excede el rate limit por IP
    o el tope global diario, lanza 429 y este handler nunca corre (no se gasta cuota Gemini).
    """
    question = req.question
    if req.table_name:
        # Si el cliente indica una tabla, se la damos al agente como pista de contexto.
        question = f"{question}\n\n(Contexto: la tabla relevante es '{req.table_name}'.)"

    try:
        result = agent_ask(question)
    except RuntimeError as exc:
        # Cuota diaria del LLM agotada u otro error no recuperable del agente.
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return AskResponse(
        answer=result.answer,
        tools_used=[
            {"name": tc.name, "args": tc.args, "ok": tc.ok} for tc in result.trace
        ],
        steps=result.steps,
    )

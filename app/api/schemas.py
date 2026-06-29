"""
schemas.py — Contratos de entrada/salida de la API (modelos Pydantic).

Qué es Pydantic y por qué se usa aquí:
  La API recibe JSON del mundo exterior, que NO es confiable (puede venir incompleto,
  con tipos equivocados, o con campos de más). Un modelo Pydantic es una "plantilla con
  reglas": FastAPI valida el JSON entrante contra ella ANTES de ejecutar tu código. Si no
  cumple, FastAPI responde 422 automáticamente con un mensaje claro, sin que escribamos
  un solo `if`. Es validación declarativa: describes la forma, no el chequeo.

Analogía:
  Es el formulario con campos obligatorios de un trámite. Si falta un campo o pones letras
  donde van números, te lo rechazan en la ventanilla (422) antes de que el trámite avance.

Convención:
  - *Request  = lo que el cliente ENVÍA.
  - *Response = lo que la API DEVUELVE.
  Para las respuestas de profile/ask usamos `dict` porque el contenido ya viene como un
  digest serializable desde las tools/el agente; no vale la pena re-tipar cada campo.
"""

from __future__ import annotations
from pydantic import BaseModel, Field


# ── Requests ────────────────────────────────────────────────────────────────────

class ConnectRequest(BaseModel):
    """Cuerpo de POST /connect."""
    path: str = Field(..., description="Ruta al archivo CSV en disco a cargar.")


class ConnectPgRequest(BaseModel):
    """Cuerpo de POST /connect-postgres."""
    table: str = Field(..., description="Nombre de la tabla en Postgres a cargar.")
    schema_name: str = Field(
        default="public",
        alias="schema",
        description="Esquema de la tabla (por defecto 'public').",
    )
    limit: int | None = Field(
        default=None,
        description="Máximo de filas a traer (opcional; por defecto todas).",
    )


class ProfileRequest(BaseModel):
    """Cuerpo de POST /profile."""
    table_name: str = Field(
        ...,
        description="Nombre de la tabla a perfilar (conocida o ya cargada).",
    )


class AskRequest(BaseModel):
    """Cuerpo de POST /ask."""
    question: str = Field(..., description="Pregunta en lenguaje natural para el agente.")
    table_name: str | None = Field(
        default=None,
        description="Tabla relevante opcional; se le pasa al agente como contexto.",
    )


# ── Responses ─────────────────────────────────────────────────────────────────--

class ToolCallInfo(BaseModel):
    """Una llamada a herramienta que hizo el agente (para trazabilidad)."""
    name: str
    args: dict
    ok: bool


class AskResponse(BaseModel):
    """Respuesta de POST /ask."""
    answer: str
    tools_used: list[ToolCallInfo]
    steps: int


class HealthResponse(BaseModel):
    """Respuesta de GET /health."""
    status: str
    service: str

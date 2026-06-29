"""
ratelimit.py — Protección anti-abuso para el endpoint /ask (el único que gasta cuota Gemini).

POR QUÉ EXISTE
  La demo es pública (cualquiera con la URL puede preguntar). Sin límites, una persona
  malintencionada podría disparar miles de /ask y vaciar la cuota gratuita de Gemini de la
  cuenta personal. Esto pone dos diques:

    1) Rate limit POR IP  → frena ráfagas de un mismo cliente (máx N preguntas por minuto).
    2) Tope GLOBAL diario → "hard cap": pase lo que pase, no se ejecutan más de M preguntas
                            por día en total. Es el dique que de verdad protege la cuota.

DECISIÓN DE DISEÑO (importante)
  El estado (contadores) vive EN MEMORIA del proceso. En Cloud Run, si hubiera varias
  instancias, cada una tendría su propio contador y el tope "global" no sería real. Por eso
  el servicio se despliega con --max-instances=1: una sola instancia ⇒ un solo contador ⇒
  el tope global es consistente. A escala (varias instancias) se usaría un store compartido
  (Redis/Firestore); para una demo de un servicio pequeño, esto es suficiente.

  No usamos librerías externas (slowapi, etc.) a propósito: el limitador es ~40 líneas de
  stdlib, fácil de leer y mantener.

CÓMO SE USA
  En main.py, /ask declara:  _rl: None = Depends(rate_limit_ask)
  FastAPI ejecuta rate_limit_ask ANTES del handler; si se excede un límite, lanza HTTP 429
  (Too Many Requests) y el handler nunca corre (no se gasta cuota).
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone

from fastapi import HTTPException, Request

# Límites configurables por variable de entorno (12-factor: ajustar sin tocar código).
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "6"))   # preguntas por IP por minuto
DAILY_ASK_CAP = int(os.getenv("DAILY_ASK_CAP", "100"))            # tope global de preguntas por día
_WINDOW_S = 60                                                    # ventana del rate limit por IP

# Estado en memoria, protegido por un lock (uvicorn atiende endpoints sync en varios hilos).
_lock = threading.Lock()
_ip_hits: dict[str, list[float]] = {}          # ip -> timestamps recientes dentro de la ventana
_daily = {"date": "", "count": 0}              # contador global del día (UTC)


def _client_ip(request: Request) -> str:
    """
    IP real del cliente. Detrás del proxy de Cloud Run, la IP del usuario viene en la
    cabecera X-Forwarded-For (el primer valor); request.client.host sería la del proxy.
    """
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit_ask(request: Request) -> None:
    """
    Dependencia de FastAPI para /ask. No devuelve nada: o pasa, o lanza HTTP 429.

    Orden de chequeo:
      1) Tope global diario (protege la cuota dura).
      2) Rate por IP (frena ráfagas de un mismo cliente).
    """
    now = time.time()
    today = datetime.now(timezone.utc).date().isoformat()
    ip = _client_ip(request)

    with _lock:
        # ── 1) Tope global diario ──────────────────────────────────────────────
        if _daily["date"] != today:           # nuevo día (UTC) → reinicia el contador
            _daily["date"] = today
            _daily["count"] = 0
        if _daily["count"] >= DAILY_ASK_CAP:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Límite diario de la demo alcanzado ({DAILY_ASK_CAP} preguntas). "
                    "Es una demo personal con cuota gratuita; vuelve a intentar mañana."
                ),
            )

        # ── 2) Rate limit por IP (ventana deslizante) ──────────────────────────
        hits = [t for t in _ip_hits.get(ip, []) if now - t < _WINDOW_S]
        if len(hits) >= RATE_LIMIT_PER_MIN:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Demasiadas preguntas seguidas (máx {RATE_LIMIT_PER_MIN}/min). "
                    "Espera unos segundos e intenta de nuevo."
                ),
            )

        # Registra esta petición (cuenta como intento, aunque luego falle: criterio conservador
        # para proteger la cuota).
        hits.append(now)
        _ip_hits[ip] = hits
        _daily["count"] += 1

        # Limpieza ligera: descarta IPs sin actividad reciente para que el dict no crezca sin fin.
        if len(_ip_hits) > 1000:
            for k in [k for k, v in _ip_hits.items() if not any(now - t < _WINDOW_S for t in v)]:
                _ip_hits.pop(k, None)

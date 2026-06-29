"""
db.py — Parámetros de conexión a Postgres, en un solo lugar.

Por qué existe (refactor del Hito 9-C):
  Antes, indexer.py, pg_loader.py y retriever.py tenían CADA UNO su propia copia de
  _conn_params(). Tres copias = tres lugares que editar si cambia algo (y se desincronizan).
  Al desplegar en la nube (Neon) necesitamos que TODAS usen SSL; centralizar evita olvidar una.

  Regla práctica: cuando la misma lógica aparece por 3ª vez, extráela. (DRY: Don't Repeat Yourself.)

Qué resuelve para el deploy:
  - Local (Docker):  DB sin SSL → DB_SSLMODE no se setea → conexión normal.
  - Neon (nube):     DB con SSL OBLIGATORIO → DB_SSLMODE=require → libpq cifra la conexión.
  El mismo código corre en ambos entornos solo cambiando variables de entorno (12-factor app).
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# Segundos para ESTABLECER la conexión antes de rendirse. Fallar rápido > colgar en silencio
# (p. ej. si Docker está apagado o el host de Neon no responde).
CONNECT_TIMEOUT_S = 5


def conn_params() -> dict:
    """
    Devuelve los parámetros de conexión a Postgres leídos del entorno (.env o vars del runtime).

    Variables reconocidas:
      DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD  → conexión básica.
      DB_SSLMODE (opcional)                            → "require" para Neon/Cloud; vacío en local.
    """
    params = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "dbname": os.getenv("DB_NAME", "gobernanza"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "postgres"),
        "connect_timeout": CONNECT_TIMEOUT_S,
    }
    # Solo agregamos sslmode si está definido: así el local (sin SSL) no se ve afectado.
    sslmode = os.getenv("DB_SSLMODE")
    if sslmode:
        params["sslmode"] = sslmode
    return params

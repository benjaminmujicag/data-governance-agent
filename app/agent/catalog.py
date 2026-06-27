"""
catalog.py — Catálogo en memoria de tablas cargadas y perfiladas.

Por qué existe:
  Las tools del agente (connect_csv, profile_table, list_catalog) necesitan
  COMPARTIR estado entre sí. Si el agente llama primero a connect_csv("clientes.csv")
  y luego a profile_table("clientes"), la segunda tool debe poder encontrar la tabla
  que cargó la primera. Este módulo es esa "memoria compartida".

Analogía:
  Es la pizarra de la oficina. Una tool escribe "ya cargué la tabla clientes (200 filas)";
  otra tool lee la pizarra para no volver a cargarla. Mientras el proceso de Python siga
  vivo, la pizarra conserva lo escrito.

Decisión de diseño (MVP, alineado con el Hito 6 "sin estado persistente"):
  El catálogo vive solo en memoria (diccionarios a nivel de módulo). Si el proceso muere,
  se pierde. Para producción se persistiría en una tabla de Postgres, pero eso es scope
  posterior. Lo importante ahora es entender el mecanismo del agente, no la persistencia.

IMPORTANTE — variables a nivel de módulo:
  En Python, un módulo se importa UNA sola vez por proceso. Por eso estos diccionarios
  son efectivamente "singletons": todas las tools que hagan `from app.agent import catalog`
  comparten exactamente los mismos diccionarios.
"""

from __future__ import annotations
from pathlib import Path

from app.connectors.base import DataTable
from app.profiler.models import ProfileReport

# Raíz del proyecto = dos niveles arriba de app/agent/catalog.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Tablas que SABEMOS que existen en el repo (nombre amigable -> ruta) ─────────
# Esto permite que el agente responda "la tabla de clientes" sin que el usuario
# tenga que pasar la ruta completa. Es el "catálogo conocido".
KNOWN_TABLES: dict[str, str] = {
    "establecimientos_salud": str(PROJECT_ROOT / "data" / "publico" / "establecimientos_salud.csv"),
    "clientes": str(PROJECT_ROOT / "data" / "catalogo" / "clientes.csv"),
    "transacciones": str(PROJECT_ROOT / "data" / "catalogo" / "transacciones.csv"),
    "productos": str(PROJECT_ROOT / "data" / "catalogo" / "productos.csv"),
    "empleados": str(PROJECT_ROOT / "data" / "catalogo" / "empleados.csv"),
    "sucursales": str(PROJECT_ROOT / "data" / "catalogo" / "sucursales.csv"),
}

# ── Estado en memoria (la "pizarra") ───────────────────────────────────────────
# Tablas ya cargadas en esta sesión: nombre -> DataTable
_LOADED: dict[str, DataTable] = {}
# Perfiles ya calculados en esta sesión: nombre -> ProfileReport
_PROFILES: dict[str, ProfileReport] = {}


def register_table(name: str, table: DataTable) -> None:
    """Guarda una tabla cargada en el catálogo bajo un nombre amigable."""
    _LOADED[name] = table


def get_table(name: str) -> DataTable | None:
    """Devuelve la tabla cargada con ese nombre, o None si no está cargada."""
    return _LOADED.get(name)


def set_profile(name: str, report: ProfileReport) -> None:
    """Cachea el perfil de una tabla para no recalcularlo en la misma sesión."""
    _PROFILES[name] = report


def get_profile(name: str) -> ProfileReport | None:
    """Devuelve el perfil cacheado de una tabla, o None si no se ha perfilado."""
    return _PROFILES.get(name)


def list_loaded() -> list[str]:
    """Nombres de las tablas cargadas en esta sesión."""
    return sorted(_LOADED.keys())


def list_profiled() -> list[str]:
    """Nombres de las tablas ya perfiladas en esta sesión."""
    return sorted(_PROFILES.keys())


def known_table_names() -> list[str]:
    """Nombres de las tablas conocidas (existan o no en memoria todavía)."""
    return sorted(KNOWN_TABLES.keys())


def resolve_known_path(name: str) -> str | None:
    """Devuelve la ruta de una tabla conocida por su nombre amigable, o None."""
    return KNOWN_TABLES.get(name)


def reset() -> None:
    """Limpia la pizarra. Útil entre tests para no arrastrar estado."""
    _LOADED.clear()
    _PROFILES.clear()

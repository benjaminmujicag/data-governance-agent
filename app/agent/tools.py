"""
tools.py — Las herramientas (tools) que el agente puede invocar.

Cada tool tiene DOS caras:
  1. La FUNCIÓN Python real (el código que se ejecuta de verdad).
  2. La DECLARACIÓN (FunctionDeclaration): la "ficha de menú" que ve el modelo.
     El modelo NO ve tu código, solo el nombre, la descripción y el schema de argumentos.
     Por eso la descripción es crítica: es lo único que le dice al modelo cuándo usar la tool.

Regla de oro de function calling:
  Las tools deben devolver datos SERIALIZABLES (dict, list, str, int, float, bool),
  porque ese resultado se le manda de vuelta al modelo como texto/JSON. NO devolver
  objetos como DataFrame o ProfileReport crudos: hay que reducirlos a un "digest".

Las 4 tools del Hito 4:
  - connect_csv(path)            → carga un CSV y lo registra en el catálogo
  - profile_table(table_name)    → perfila una tabla (la carga si hace falta)
  - search_governance(query, k)  → busca en el corpus RAG y devuelve fragmentos + citas
  - list_catalog()               → lista tablas cargadas y disponibles
"""

from __future__ import annotations
from pathlib import Path

from google.genai import types

from app.connectors.csv_loader import load_csv
from app.profiler.profiler import profile
from app.profiler.models import ProfileReport
from app.rag.retriever import retrieve
from app.agent import catalog


# ── Helpers de serialización ───────────────────────────────────────────────────

def _report_to_digest(report: ProfileReport) -> dict:
    """
    Reduce un ProfileReport a un dict compacto para mandárselo al modelo.

    Por qué un digest y no el reporte completo:
      El reporte tiene mucha info; mandarla toda gasta tokens y mete ruido.
      El modelo necesita lo accionable: forma, PII, flags de calidad y, por columna,
      tipo inferido, % de nulos y banderas. Con eso responde preguntas de gobernanza.
    """
    return {
        "table_name": report.table_name,
        "source": report.source,
        "shape": {"filas": report.shape[0], "columnas": report.shape[1]},
        "filas_duplicadas": report.row_duplicate_count,
        "columnas_pii": report.pii_columns,
        "flags_calidad_tabla": report.quality_flags,
        "resumen": report.summary,
        "columnas": [
            {
                "nombre": c.name,
                "tipo_inferido": c.dtype_inferred,
                "nulos_pct": c.null_pct,
                "nulos_efectivos_pct": c.effective_null_pct,
                "pii": c.pii_type,
                "flags_calidad": c.quality_flags,
            }
            for c in report.columns
        ],
    }


# ── Implementaciones de las tools ───────────────────────────────────────────────

def connect_csv(path: str) -> dict:
    """
    Carga un CSV desde una ruta y lo registra en el catálogo en memoria.

    Devuelve un resumen (NO el DataFrame completo): nombre, origen, forma,
    columnas y advertencias de carga.
    """
    table = load_csv(path)
    name = Path(path).stem  # nombre de archivo sin extensión = nombre amigable
    catalog.register_table(name, table)
    return {
        "table_name": name,
        "source": table.source,
        "shape": {
            "filas": table.df.shape[0],
            "columnas": table.df.shape[1],
        },
        "columnas": list(table.df.columns),
        "advertencias": table.metadata.get("warnings", []),
    }


def profile_table(table_name: str) -> dict:
    """
    Perfila una tabla y devuelve un digest con esquema, nulos, PII y flags de calidad.

    Resolución de la tabla (en orden):
      1. Si ya fue perfilada en esta sesión → devuelve el perfil cacheado.
      2. Si está cargada pero no perfilada → la perfila.
      3. Si es una tabla CONOCIDA no cargada → la carga y la perfila.
      4. Si no se reconoce → error con la lista de tablas disponibles.
    """
    cached = catalog.get_profile(table_name)
    if cached is not None:
        return _report_to_digest(cached)

    table = catalog.get_table(table_name)
    if table is None:
        known_path = catalog.resolve_known_path(table_name)
        if known_path is None:
            return {
                "error": f"No conozco la tabla '{table_name}'.",
                "tablas_disponibles": catalog.known_table_names(),
            }
        table = load_csv(known_path)
        catalog.register_table(table_name, table)

    report = profile(table, table_name=table_name)
    catalog.set_profile(table_name, report)
    return _report_to_digest(report)


def search_governance(query: str, k: int = 5) -> dict:
    """
    Busca en el corpus de gobernanza (políticas internas + Ley 21.719) los
    fragmentos más relevantes para la query y los devuelve con su cita.

    Cada fragmento incluye doc_name (la cita verificable), chunk_id, score y texto.
    """
    chunks = retrieve(query, k=k, min_score=0.3)
    return {
        "n_resultados": len(chunks),
        "fragmentos": [
            {
                "documento": c.doc_name,
                "fragmento_id": c.chunk_id,
                "relevancia": c.score,
                "texto": c.text,
            }
            for c in chunks
        ],
    }


def list_catalog() -> dict:
    """
    Lista las tablas cargadas/perfiladas en esta sesión y las tablas conocidas
    que se pueden perfilar bajo demanda.
    """
    return {
        "tablas_cargadas": catalog.list_loaded(),
        "tablas_perfiladas": catalog.list_profiled(),
        "tablas_disponibles": catalog.known_table_names(),
    }


# ── Declaraciones (lo que ve el modelo) ─────────────────────────────────────────
# El schema sigue el estándar JSON Schema. Es la única información que el modelo
# usa para decidir cuándo y cómo llamar cada tool. Descripciones claras = mejor agente.

_FUNCTION_DECLARATIONS = [
    types.FunctionDeclaration(
        name="connect_csv",
        description=(
            "Carga un archivo CSV desde una ruta del disco y lo registra en el catálogo. "
            "Úsala cuando el usuario quiera conectar/ingerir una tabla nueva indicando su ruta."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Ruta al archivo CSV en disco.",
                },
            },
            "required": ["path"],
        },
    ),
    types.FunctionDeclaration(
        name="profile_table",
        description=(
            "Perfila una tabla y devuelve su esquema, porcentaje de nulos, columnas con PII "
            "y flags de calidad (duplicados, fechas como texto, columnas vacías, etc.). "
            "Úsala para responder sobre CALIDAD de datos o sobre qué columnas contienen PII. "
            "Acepta nombres conocidos como: establecimientos_salud, clientes, transacciones, "
            "productos, empleados, sucursales."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": "Nombre amigable de la tabla a perfilar.",
                },
            },
            "required": ["table_name"],
        },
    ),
    types.FunctionDeclaration(
        name="search_governance",
        description=(
            "Busca en el corpus de gobernanza (políticas internas de datos y la Ley 21.719 "
            "de Protección de Datos de Chile) los fragmentos más relevantes para una pregunta. "
            "Úsala para responder sobre políticas, normas, plazos de retención, clasificación "
            "de datos, acceso, o lo que dice la ley. Devuelve fragmentos con su documento fuente."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Pregunta o tema a buscar en el corpus de gobernanza.",
                },
                "k": {
                    "type": "integer",
                    "description": "Número de fragmentos a recuperar (por defecto 5).",
                },
            },
            "required": ["query"],
        },
    ),
    types.FunctionDeclaration(
        name="list_catalog",
        description=(
            "Lista las tablas cargadas y perfiladas en la sesión, y las tablas disponibles "
            "que se pueden perfilar. Úsala cuando el usuario pregunte qué tablas hay."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
]

# La Tool agrupa todas las declaraciones que se le pasan al modelo en la config.
TOOL = types.Tool(function_declarations=_FUNCTION_DECLARATIONS)

# DISPATCH conecta el NOMBRE que pide el modelo con la FUNCIÓN real a ejecutar.
# El agente hace: DISPATCH[nombre_pedido](**argumentos).
DISPATCH = {
    "connect_csv": connect_csv,
    "profile_table": profile_table,
    "search_governance": search_governance,
    "list_catalog": list_catalog,
}

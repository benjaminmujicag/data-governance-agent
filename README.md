# Agente de Gobernanza de Datos (RAG + Conectores)

Agente de IA al que le **conectas tablas vengan de donde vengan**; las **perfila** automáticamente
(esquema, calidad, posibles datos personales/PII), las **cataloga**, y responde preguntas de gobernanza
en lenguaje natural **citando la fuente exacta** (la política, la ley o la columna).

## ¿Por qué este proyecto?
En bancos y retail, los analistas pierden horas entendiendo de dónde viene un dato, si es confiable y si
pueden usarlo sin violar normativa. Este agente ingiere la tabla, la perfila y responde con citas, reduciendo
tiempo y riesgo de incumplimiento.

Todos los datos del repo son **sintéticos o públicos** (tablas generadas, políticas de ejemplo y un resumen
de la Ley 21.719 de Chile). No contiene información real ni sensible.

## Stack
- **Python** + **FastAPI** (API REST)
- **PostgreSQL + pgVector** (base vectorial, corre en Docker, local)
- **Gemini** como LLM/embeddings (proveedor **intercambiable** vía variable de entorno)
- **pandas** para perfilado de datos
- Agente con **function calling** (tools) y loop de razonamiento construido desde cero (sin frameworks de agentes)

## Arquitectura (alto nivel)
```
Fuentes (CSV) → Conectores → Perfilador (esquema, nulos, PII)
                                   │
   Corpus de gobernanza ──► chunking + embeddings ──► pgVector
   (políticas + ley)                                     ▲
                                                         │ retrieval top-k
        Pregunta ──► API (FastAPI) ──► Agente (LLM + tools) ──► Respuesta + citas
```
El agente decide qué herramienta usar según la pregunta: perfilar una tabla, buscar en el catálogo, o
consultar el corpus de gobernanza (RAG con citas).

## Componentes
- `app/connectors/` — conector CSV con autodetección de encoding y separador (interfaz `DataTable`).
- `app/profiler/` — perfilado de calidad: nulos reales y semánticos, tipos inferidos, detección heurística de PII.
- `app/rag/` — chunking, embeddings (Gemini), indexación en pgVector y recuperación con similitud coseno.
- `app/agent/` — herramientas (tools), catálogo y loop de function calling con reintentos ante rate limits.
- `app/api/` — API FastAPI que expone el agente como servicio HTTP.

## Cómo correr
1. Copia `.env.example` a `.env` y completa tu `GOOGLE_API_KEY`.
2. Levanta la base vectorial: `docker compose up -d`
3. Crea el entorno e instala dependencias:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows (en Linux/Mac: source .venv/bin/activate)
   pip install -r requirements.txt
   ```
4. Verifica el entorno: `python scripts/check_setup.py`
5. Indexa el corpus de gobernanza en pgVector: `python scripts/indexar_corpus.py`
6. Levanta la API: `uvicorn app.api.main:app --reload` → documentación interactiva en `http://localhost:8000/docs`

## Tests
- `python scripts/test_conector.py` — conector CSV
- `python scripts/test_perfilador.py` — perfilador
- `python scripts/test_rag.py` — recuperación del RAG
- `python scripts/test_agente.py` — agente de extremo a extremo (function calling)
- `python scripts/test_api.py` — API (requiere el servidor levantado)

## Estado
En desarrollo activo. Implementado: conectores CSV, perfilador con detección de PII, RAG con citas sobre
pgVector, agente con function calling y API REST. Pendiente: UI (Streamlit), evaluación (golden set +
LLM-as-Judge) y empaquetado en Docker.

"""
app.py — UI mínima (Streamlit) del Agente de Gobernanza de Datos (Hito 6).

Qué es esto:
  Una pantalla web para DEMOSTRAR el agente sin escribir JSON a mano. Permite:
    - elegir una tabla conocida o subir un CSV nuevo,
    - perfilarla (esquema, nulos, PII, flags de calidad),
    - chatear con el agente y ver su respuesta con citas y las tools que usó.

Decisión de arquitectura (Hito 6):
  Esta UI NO contiene lógica de negocio. Es un CLIENTE de la API (Hito 5): habla con
  uvicorn por HTTP usando httpx, igual que lo hace scripts/test_api.py o el navegador.
  Ventaja: backend y frontend quedan desacoplados (podrías cambiar Streamlit por React
  sin tocar el agente). Costo: hay que tener DOS procesos vivos (uvicorn + streamlit).

Modelo mental de Streamlit (importante para leer este archivo):
  Streamlit re-ejecuta TODO este script de arriba a abajo cada vez que el usuario
  interactúa (un clic, escribir en un input). Por eso, para que la app "recuerde" cosas
  entre re-ejecuciones (el historial del chat, el último perfil), se usa st.session_state,
  que es un diccionario que sobrevive a esas re-ejecuciones.

Cómo levantarla (necesita la API corriendo en otra terminal):
  Terminal 1:  uvicorn app.api.main:app --reload
  Terminal 2:  streamlit run app/ui/app.py
  (Recuerda Docker abierto + corpus indexado para que el chat con políticas funcione,
   y LLM_MODEL=gemini-2.5-flash en el entorno por el tema de cuota.)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import httpx
import streamlit as st

# URL donde escucha la API (uvicorn). Configurable desde la barra lateral.
# Default segun el entorno: en local es localhost; dentro de Docker el contenedor 'ui'
# recibe API_URL=http://api:8000 (la API se alcanza por el NOMBRE del servicio, no localhost).
DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")
# El agente puede tardar (varias llamadas al LLM + reintentos con backoff).
TIMEOUT = 120.0


# ══════════════════════════════════════════════════════════════════════════════
#  CAPA 1 — Cliente de la API (las funciones que hablan HTTP con uvicorn)
#  Cada función traduce un endpoint a una llamada Python. Si la API responde con
#  un código de error (404/422/503), httpx lo convierte en excepción con
#  raise_for_status(); la capa de presentación decide cómo mostrarlo.
# ══════════════════════════════════════════════════════════════════════════════

def api_health(base_url: str) -> bool:
    """True si la API responde en /health. Se usa para avisar si uvicorn no está vivo."""
    try:
        r = httpx.get(f"{base_url}/health", timeout=5.0)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except httpx.HTTPError:
        return False


def api_catalog(base_url: str) -> dict:
    """GET /catalog → tablas cargadas, perfiladas y disponibles."""
    r = httpx.get(f"{base_url}/catalog", timeout=10.0)
    r.raise_for_status()
    return r.json()


def api_connect(base_url: str, path: str) -> dict:
    """POST /connect → carga un CSV (por su ruta en disco) y lo registra en el catálogo."""
    r = httpx.post(f"{base_url}/connect", json={"path": path}, timeout=30.0)
    r.raise_for_status()
    return r.json()


def api_profile(base_url: str, table_name: str) -> dict:
    """POST /profile → digest del perfilado (esquema, nulos, PII, flags). No usa LLM."""
    r = httpx.post(f"{base_url}/profile", json={"table_name": table_name}, timeout=60.0)
    r.raise_for_status()
    return r.json()


def api_ask(base_url: str, question: str, table_name: str | None) -> dict:
    """POST /ask → el agente responde (SÍ usa LLM). Devuelve answer + tools_used + steps."""
    payload: dict = {"question": question}
    if table_name:
        payload["table_name"] = table_name
    r = httpx.post(f"{base_url}/ask", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def guardar_upload_a_disco(uploaded_file) -> str:
    """
    Puente entre el file_uploader (que da BYTES) y /connect (que espera una RUTA).

    Por qué existe: la API recibe la ruta de un archivo en el disco del SERVIDOR. En la
    demo, UI y API corren en la misma máquina, así que escribir el CSV subido en una
    carpeta temporal y mandar esa ruta funciona. (En producción con máquinas separadas,
    el endpoint debería recibir los bytes directamente; queda como deuda consciente.)
    """
    destino = Path(tempfile.gettempdir()) / uploaded_file.name
    destino.write_bytes(uploaded_file.getbuffer())
    return str(destino)


# ══════════════════════════════════════════════════════════════════════════════
#  CAPA 2 — Presentación (lo que dibuja Streamlit)
# ══════════════════════════════════════════════════════════════════════════════

def init_state() -> None:
    """Crea las claves de session_state la primera vez (sobreviven a las re-ejecuciones)."""
    st.session_state.setdefault("messages", [])      # historial del chat: [{role, content, meta}]
    st.session_state.setdefault("profile", None)     # último digest de perfilado mostrado
    st.session_state.setdefault("profile_table", None)  # nombre de la tabla perfilada


def render_sidebar() -> tuple[str, str | None]:
    """
    Dibuja la barra lateral y devuelve (base_url, tabla_seleccionada).

    Permite: configurar la URL de la API, ver si está viva, elegir una tabla conocida
    o subir un CSV nuevo, y disparar el perfilado.
    """
    with st.sidebar:
        st.header("Configuración")
        base_url = st.text_input("URL de la API", value=DEFAULT_API_URL)

        viva = api_health(base_url)
        if viva:
            st.success("API conectada")
        else:
            st.error("API no responde. Levanta: uvicorn app.api.main:app --reload")
            return base_url, None

        st.divider()
        st.header("Fuente de datos")

        # Dos formas de elegir tabla: del catálogo conocido, o subiendo un CSV nuevo.
        try:
            catalogo = api_catalog(base_url)
            disponibles = catalogo.get("tablas_disponibles", [])
        except httpx.HTTPError:
            disponibles = []

        modo = st.radio("¿De dónde sacamos la tabla?", ["Tabla conocida", "Subir CSV nuevo"])
        tabla_seleccionada: str | None = None

        if modo == "Tabla conocida":
            tabla_seleccionada = st.selectbox("Tabla", disponibles) if disponibles else None
        else:
            archivo = st.file_uploader("Sube un archivo CSV", type=["csv"])
            if archivo is not None and st.button("Conectar CSV"):
                try:
                    ruta = guardar_upload_a_disco(archivo)
                    res = api_connect(base_url, ruta)
                    st.success(f"Cargada '{res['table_name']}' ({res['shape']['filas']} filas)")
                    # El nombre amigable que registra la API es el del archivo sin extensión.
                    tabla_seleccionada = res["table_name"]
                except httpx.HTTPError as exc:
                    st.error(f"No se pudo conectar el CSV: {exc}")

        st.divider()
        if tabla_seleccionada and st.button("Perfilar tabla", type="primary"):
            try:
                st.session_state["profile"] = api_profile(base_url, tabla_seleccionada)
                st.session_state["profile_table"] = tabla_seleccionada
            except httpx.HTTPError as exc:
                st.error(f"No se pudo perfilar: {exc}")

        return base_url, tabla_seleccionada


def render_perfil() -> None:
    """Panel izquierdo: muestra el digest del perfilador (o una pista si no hay nada)."""
    st.subheader("Perfilado de la tabla")
    perfil = st.session_state.get("profile")

    if not perfil:
        st.info("Elige una tabla en la barra lateral y pulsa 'Perfilar tabla'.")
        return

    forma = perfil.get("shape", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas", forma.get("filas", "-"))
    c2.metric("Columnas", forma.get("columnas", "-"))
    c3.metric("Duplicadas", perfil.get("filas_duplicadas", "-"))
    c4.metric("Columnas con PII", len(perfil.get("columnas_pii", [])))

    pii = perfil.get("columnas_pii", [])
    if pii:
        st.warning("PII detectada en: " + ", ".join(pii))

    flags = perfil.get("flags_calidad_tabla", [])
    if flags:
        st.write("**Flags de calidad (tabla):**")
        for f in flags:
            st.write(f"- {f}")

    # Tabla por columna: st.dataframe acepta una lista de dicts directamente.
    columnas = perfil.get("columnas", [])
    if columnas:
        st.write("**Detalle por columna:**")
        st.dataframe(columnas, use_container_width=True, hide_index=True)


def render_chat(base_url: str) -> None:
    """Panel derecho: historial del chat + input de pregunta para el agente."""
    st.subheader("Chat con el agente")

    # Pintamos el historial guardado en session_state.
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            meta = msg.get("meta")
            if meta:
                tools = ", ".join(meta.get("tools", [])) or "ninguna"
                st.caption(f"Tools usadas: {tools} · pasos: {meta.get('steps', '-')}")

    # Input de pregunta. st.chat_input se ancla al fondo del contenedor.
    pregunta = st.chat_input("Pregunta sobre calidad, PII, políticas o la Ley 21.719…")
    if not pregunta:
        return

    st.session_state["messages"].append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("El agente está pensando…"):
            try:
                # Si hay una tabla perfilada, se la pasamos como pista de contexto.
                res = api_ask(base_url, pregunta, st.session_state.get("profile_table"))
                respuesta = res.get("answer", "(sin respuesta)")
                tools = [t["name"] for t in res.get("tools_used", [])]
                steps = res.get("steps", "-")
            except httpx.HTTPStatusError as exc:
                # 503 = cuota del LLM agotada; 404 = tabla no encontrada; etc.
                respuesta = f"Error de la API ({exc.response.status_code}): {exc.response.text}"
                tools, steps = [], "-"
            except httpx.HTTPError as exc:
                respuesta = f"No se pudo contactar la API: {exc}"
                tools, steps = [], "-"

        st.markdown(respuesta)
        st.caption(f"Tools usadas: {', '.join(tools) or 'ninguna'} · pasos: {steps}")

    st.session_state["messages"].append(
        {"role": "assistant", "content": respuesta, "meta": {"tools": tools, "steps": steps}}
    )


def main() -> None:
    st.set_page_config(page_title="Agente de Gobernanza de Datos", layout="wide")
    st.title("Agente de Gobernanza de Datos")
    st.caption("Conecta una tabla, perfílala y pregúntale al agente. Las respuestas citan su fuente.")

    init_state()
    base_url, _ = render_sidebar()

    col_perfil, col_chat = st.columns(2)
    with col_perfil:
        render_perfil()
    with col_chat:
        render_chat(base_url)


if __name__ == "__main__":
    main()

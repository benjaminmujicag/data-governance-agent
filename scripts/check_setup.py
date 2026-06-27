"""
Verificación del entorno (Hito 0).

Comprueba dos cosas antes de empezar a construir:
  1) Que la base Postgres + pgVector responde y sabe operar con vectores.
  2) Que la API de Gemini está accesible (si hay GOOGLE_API_KEY).

Ejecuta:  python scripts/check_setup.py
(asegúrate de tener el entorno activado y las dependencias instaladas)
"""
import os
import sys

from dotenv import load_dotenv

load_dotenv()  # carga las variables desde el archivo .env


def check_database() -> bool:
    """Conecta a Postgres, activa pgVector y hace una búsqueda por similitud de juguete."""
    try:
        import psycopg
        from pgvector.psycopg import register_vector
    except ImportError:
        print("[DB] Falta instalar dependencias: pip install -r requirements.txt")
        return False

    try:
        conn = psycopg.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            dbname=os.getenv("DB_NAME", "gobernanza"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "postgres"),
            connect_timeout=5,
        )
    except Exception as e:
        print(f"[DB] No pude conectar. ¿Está corriendo 'docker compose up -d'?\n     Detalle: {e}")
        return False

    with conn:
        # pgVector se activa una vez por base de datos; habilita el tipo 'vector'.
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        register_vector(conn)

        # Tabla temporal: vive solo durante esta conexión, ideal para una prueba.
        conn.execute("CREATE TEMP TABLE prueba (id int, emb vector(3));")
        conn.execute("INSERT INTO prueba (id, emb) VALUES (1, '[1,1,1]'), (2, '[9,9,9]');")

        # Buscamos el vecino más cercano a [1,1,2] usando distancia coseno (<=>).
        fila = conn.execute(
            "SELECT id FROM prueba ORDER BY emb <=> '[1,1,2]' LIMIT 1;"
        ).fetchone()

    print(f"[DB] OK · pgVector funciona. Vecino más cercano a [1,1,2] = fila id={fila[0]} (esperado: 1).")
    return True


def check_gemini() -> bool:
    """Pide un embedding de prueba a Gemini para confirmar que la API key sirve."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.startswith("tu_api_key"):
        print("[Gemini] Sin GOOGLE_API_KEY válida en .env (lo dejamos para cuando la tengas).")
        return False

    try:
        from google import genai
    except ImportError:
        print("[Gemini] Falta instalar dependencias: pip install -r requirements.txt")
        return False

    try:
        client = genai.Client(api_key=api_key)
        model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
        result = client.models.embed_content(model=model, contents="hola gobernanza de datos")
        dim = len(result.embeddings[0].values)
        print(f"[Gemini] OK · embedding generado con '{model}' · dimensión={dim}.")
        return True
    except Exception as e:
        print(f"[Gemini] No pude generar embedding. Revisa la API key/modelo.\n         Detalle: {e}")
        return False


if __name__ == "__main__":
    print("== Verificación de entorno (Hito 0) ==")
    db_ok = check_database()
    gemini_ok = check_gemini()
    print("\nResumen:")
    print(f"  Base vectorial : {'OK' if db_ok else 'PENDIENTE'}")
    print(f"  Gemini API     : {'OK' if gemini_ok else 'PENDIENTE'}")
    sys.exit(0 if db_ok else 1)

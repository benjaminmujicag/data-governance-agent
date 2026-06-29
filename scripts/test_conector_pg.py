"""
Smoke test del conector Postgres (Hito 9-D).

Idea del test (demuestra "ingesta multi-fuente"):
    1. SIEMBRA una tabla de prueba en Postgres con datos sucios a proposito
       (PII en rut/email/telefono, un NULL y una fila duplicada).
    2. La LEE con el conector nuevo load_postgres() -> DataTable.
    3. La PERFILA con el MISMO perfilador del CSV -> ProfileReport.
       Esto prueba el valor de la abstraccion DataTable: el perfilador no sabe
       ni le importa que el dato venga de Postgres en vez de un archivo.
    4. Limpia la tabla de prueba al final (la borra), pase lo que pase.

Precondicion: Docker Desktop abierto y `docker compose up -d` (la DB en localhost:5432).
Este test NO usa Gemini -> NO consume cuota del free tier.

Como saber si salio bien: todas las lineas dicen [OK] y el script termina con codigo 0.

Ejecucion:
    python scripts/test_conector_pg.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import psycopg

from app.connectors.pg_loader import load_postgres, _conn_params, DatabaseUnavailableError
from app.connectors.base import DataTable
from app.profiler.profiler import profile

PASS = "[OK]  "
FAIL = "[FAIL]"
errors = 0

DEMO_TABLE = "demo_clientes_pg"


def check(descripcion: str, condicion: bool, detalle: str = "") -> None:
    global errors
    if condicion:
        print(f"  {PASS} {descripcion}")
    else:
        print(f"  {FAIL} {descripcion}")
        if detalle:
            print(f"         Detalle: {detalle}")
        errors += 1


def sembrar_tabla() -> None:
    """
    Crea la tabla de prueba con datos sucios a proposito.

    Por que sembramos en vez de usar una tabla real: nuestra DB de Docker solo tiene
    el corpus de gobernanza (governance_chunks), no tablas de negocio. Para probar el
    conector necesitamos una tabla que leer; la creamos aqui y la borramos al final.
    """
    conn = psycopg.connect(**_conn_params())
    try:
        conn.execute(f"DROP TABLE IF EXISTS {DEMO_TABLE};")
        conn.execute(f"""
            CREATE TABLE {DEMO_TABLE} (
                rut       TEXT,
                nombre    TEXT,
                email     TEXT,
                telefono  TEXT,
                edad      INTEGER,
                salario   INTEGER,
                ciudad    TEXT
            );
        """)
        filas = [
            ("12.345.678-9", "Ana Soto",    "ana.soto@example.com",   "+56912345678", 34, 1500000, "Santiago"),
            ("9.876.543-2",  "Luis Pena",   "luis.pena@example.com",  "+56987654321", 41, 2200000, "Valparaiso"),
            ("11.222.333-4", "Marta Rivas", "marta.rivas@example.com", None,          29, 1800000, "Concepcion"),
            ("7.654.321-0",  "Jose Vidal",  "jose.vidal@example.com", "+56955551234", 52, 3100000, "Santiago"),
            # Fila duplicada EXACTA de la primera (para probar deteccion de duplicados)
            ("12.345.678-9", "Ana Soto",    "ana.soto@example.com",   "+56912345678", 34, 1500000, "Santiago"),
        ]
        conn.cursor().executemany(
            f"INSERT INTO {DEMO_TABLE} (rut, nombre, email, telefono, edad, salario, ciudad) "
            f"VALUES (%s, %s, %s, %s, %s, %s, %s)",
            filas,
        )
        conn.commit()
    finally:
        conn.close()


def borrar_tabla() -> None:
    """Borra la tabla de prueba (limpieza). No falla si ya no existe."""
    try:
        conn = psycopg.connect(**_conn_params())
        try:
            conn.execute(f"DROP TABLE IF EXISTS {DEMO_TABLE};")
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        print(f"  (aviso) no se pudo borrar la tabla de prueba: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  SMOKE TEST — Conector Postgres")
print("=" * 60)

# Verificacion temprana: ¿esta la DB arriba? Si no, mensaje claro y salimos.
try:
    sembrar_tabla()
except DatabaseUnavailableError as exc:
    print(f"\n  {FAIL} La base de datos no responde.")
    print(f"         {exc}")
    print("\n  Levanta la DB con:  docker compose up -d   (Docker Desktop abierto)")
    sys.exit(1)
except psycopg.OperationalError as exc:
    print(f"\n  {FAIL} No se pudo conectar/sembrar en Postgres: {exc}")
    print("\n  Levanta la DB con:  docker compose up -d   (Docker Desktop abierto)")
    sys.exit(1)

try:
    # ── Test 1: leer la tabla completa ────────────────────────────────────────
    print()
    print(f"  [TEST 1] Leer tabla completa '{DEMO_TABLE}'")

    tabla = load_postgres(table=DEMO_TABLE)
    check("Devuelve un DataTable",        isinstance(tabla, DataTable))
    check("source == 'postgres'",         tabla.source == "postgres",
          f"source real: {tabla.source!r}")
    check("path_or_table == 'public.demo_clientes_pg'",
          tabla.path_or_table == f"public.{DEMO_TABLE}",
          f"valor real: {tabla.path_or_table!r}")
    check("5 filas x 7 columnas",         tabla.df.shape == (5, 7),
          f"shape real: {tabla.df.shape}")
    check("metadata tiene 'server_version'", "server_version" in tabla.metadata)
    check("metadata tiene 'warnings'",       "warnings" in tabla.metadata)

    # ── Test 2: el MISMO perfilador detecta la PII ────────────────────────────
    print()
    print("  [TEST 2] El perfilador (mismo del CSV) procesa la tabla de Postgres")

    report = profile(tabla, table_name="demo_clientes_pg")
    check("ProfileReport.source == 'postgres'", report.source == "postgres",
          f"source real: {report.source!r}")
    check("Detecta PII en 'rut'",      "rut" in report.pii_columns,
          f"pii_columns: {report.pii_columns}")
    check("Detecta PII en 'email'",    "email" in report.pii_columns,
          f"pii_columns: {report.pii_columns}")
    check("Detecta PII en 'telefono'", "telefono" in report.pii_columns,
          f"pii_columns: {report.pii_columns}")
    check("Detecta la fila duplicada", report.row_duplicate_count == 1,
          f"duplicados detectados: {report.row_duplicate_count}")

    # ── Test 3: ruta por consulta (query) + parametro limit ───────────────────
    print()
    print("  [TEST 3] Lectura por query y por limit")

    t_query = load_postgres(query=f"SELECT rut, email FROM {DEMO_TABLE}")
    check("query devuelve 2 columnas", list(t_query.df.columns) == ["rut", "email"],
          f"columnas: {list(t_query.df.columns)}")
    check("path_or_table == '<query>'", t_query.path_or_table == "<query>")

    t_limit = load_postgres(table=DEMO_TABLE, limit=2)
    check("limit=2 trae exactamente 2 filas", len(t_limit.df) == 2,
          f"filas: {len(t_limit.df)}")

    # ── Test 4: validacion de argumentos ──────────────────────────────────────
    print()
    print("  [TEST 4] Validacion: ni table ni query -> ValueError")

    try:
        load_postgres()
        check("Debe lanzar ValueError", False, "no lanzo excepcion")
    except ValueError:
        check("Lanza ValueError correctamente", True)
    except Exception as exc:  # noqa: BLE001
        check("Debe lanzar ValueError", False, f"lanzo {type(exc).__name__}: {exc}")

finally:
    borrar_tabla()

# ── Resultado final ───────────────────────────────────────────────────────────
print()
print("=" * 60)
if errors == 0:
    print("  RESULTADO: TODOS LOS CHECKS PASARON")
    print("  El conector Postgres ingiere y perfila igual que el CSV.")
else:
    print(f"  RESULTADO: {errors} CHECK(S) FALLARON — revisar arriba")
    sys.exit(1)
print("=" * 60)
print()

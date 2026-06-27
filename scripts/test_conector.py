"""
Smoke test del conector CSV (Hito 1).

Un "smoke test" no prueba cada detalle: verifica que lo fundamental
funciona (como encender el motor y ver que no humea). Si este script
corre sin errores, el conector está listo para el Hito 2.

Qué verifica:
    1. Carga el dataset sucio real (MINSAL) y el catálogo sintético.
    2. Que DataTable tiene .df, .source, .path_or_table y .metadata.
    3. Que la auto-detección de encoding y separador fue correcta.
    4. Que las dimensiones son las esperadas.
    5. Que puedes pasar parámetros opcionales (nrows, encoding manual).

Cómo saber si salió bien: todas las líneas dicen [OK].
Cómo saber si falló:      aparece [FAIL] con el mensaje de error.

Ejecución:
    python scripts/test_conector.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.connectors.csv_loader import load_csv
from app.connectors.base import DataTable

PASS = "[OK]  "
FAIL = "[FAIL]"
errors = 0


def check(descripcion: str, condicion: bool, detalle: str = "") -> None:
    global errors
    if condicion:
        print(f"  {PASS} {descripcion}")
    else:
        print(f"  {FAIL} {descripcion}")
        if detalle:
            print(f"         Detalle: {detalle}")
        errors += 1


# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  SMOKE TEST — Conector CSV")
print("=" * 60)

# ── Test 1: dataset sucio real ────────────────────────────────────────────────
print()
print("  [TEST 1] Dataset real — establecimientos_salud.csv")

path_real = ROOT / "data" / "publico" / "establecimientos_salud.csv"
tabla_real = load_csv(path_real)

check("Devuelve un DataTable",           isinstance(tabla_real, DataTable))
check("source == 'csv'",                 tabla_real.source == "csv")
check("DataFrame no vacío",              len(tabla_real.df) > 0)
check("Filas esperadas (~5707)",         len(tabla_real.df) == 5707,
      f"encontradas: {len(tabla_real.df)}")
check("Columnas esperadas (33)",         len(tabla_real.df.columns) == 33,
      f"encontradas: {len(tabla_real.df.columns)}")
check("Auto-detecto separador ';'",      tabla_real.metadata["separator"] == ";",
      f"detectado: {tabla_real.metadata['separator']!r}")
check("Auto-detecto encoding utf-8",     tabla_real.metadata["encoding"] == "utf-8",
      f"detectado: {tabla_real.metadata['encoding']}")
check("metadata tiene 'shape'",          "shape" in tabla_real.metadata)
check("metadata tiene 'null_pct_global'", "null_pct_global" in tabla_real.metadata)
check("metadata tiene 'warnings'",        "warnings" in tabla_real.metadata)

# ── Test 2: tablas sintéticas (catálogo) ──────────────────────────────────────
print()
print("  [TEST 2] Tablas sinteticas del catalogo")

esperados = {
    "clientes.csv":      (200, 9),
    "transacciones.csv": (500, 8),
    "productos.csv":     (60,  7),
    "empleados.csv":     (50,  9),
    "sucursales.csv":    (8,   9),
}

for nombre, (filas, cols) in esperados.items():
    path = ROOT / "data" / "catalogo" / nombre
    t = load_csv(path)
    check(
        f"{nombre}: {filas} filas x {cols} cols",
        t.df.shape == (filas, cols),
        f"shape real: {t.df.shape}",
    )
    check(
        f"{nombre}: separador auto-detectado ','",
        t.metadata["separator"] == ",",
        f"detectado: {t.metadata['separator']!r}",
    )

# ── Test 3: parámetro nrows ───────────────────────────────────────────────────
print()
print("  [TEST 3] Parametro opcional nrows=10")

tabla_10 = load_csv(path_real, nrows=10)
check("Con nrows=10 se cargan exactamente 10 filas",
      len(tabla_10.df) == 10,
      f"encontradas: {len(tabla_10.df)}")

# ── Test 4: archivo que no existe ─────────────────────────────────────────────
print()
print("  [TEST 4] Archivo inexistente lanza FileNotFoundError")

try:
    load_csv(ROOT / "data" / "no_existe.csv")
    check("Debe lanzar FileNotFoundError", False, "no lanzó excepción")
except FileNotFoundError:
    check("Lanza FileNotFoundError correctamente", True)
except Exception as exc:
    check("Debe lanzar FileNotFoundError", False, f"lanzó {type(exc).__name__}: {exc}")

# ── Resultado final ───────────────────────────────────────────────────────────
print()
print("=" * 60)
if errors == 0:
    print("  RESULTADO: TODOS LOS CHECKS PASARON")
    print("  El conector CSV esta listo para el Hito 2 (perfilador).")
else:
    print(f"  RESULTADO: {errors} CHECK(S) FALLARON — revisar arriba")
    sys.exit(1)
print("=" * 60)
print()

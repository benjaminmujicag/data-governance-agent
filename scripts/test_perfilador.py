"""
Prueba el perfilador automático sobre el dataset real y una tabla sintética.

Ejecución:
    python scripts/test_perfilador.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.connectors.csv_loader import load_csv
from app.profiler.profiler import profile

PASS = "[OK]  "
FAIL = "[FAIL]"
errors = 0


def check(desc: str, cond: bool, detalle: str = "") -> None:
    global errors
    if cond:
        print(f"  {PASS} {desc}")
    else:
        print(f"  {FAIL} {desc}" + (f"  ({detalle})" if detalle else ""))
        errors += 1


def separador(titulo: str) -> None:
    print()
    print("=" * 65)
    print(f"  {titulo}")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
separador("TEST 1 — Dataset real: establecimientos_salud.csv")

tabla_real = load_csv(ROOT / "data" / "publico" / "establecimientos_salud.csv")
rpt = profile(tabla_real, table_name="establecimientos_salud")

print(f"\n  {rpt}")
print(f"\n  Resumen:")
for k, v in rpt.summary.items():
    print(f"    {k:<35} {v}")

print()

# Checks de estructura
check("Devuelve ProfileReport",              rpt.__class__.__name__ == "ProfileReport")
check("shape correcta (5707x33)",            rpt.shape == (5707, 33),
      f"got {rpt.shape}")
check("columns tiene 33 ColumnProfile",      len(rpt.columns) == 33)
check("Detecta columnas con nulos",          rpt.summary["columnas_con_nulos"] > 0)
check("Detecta nulos semanticos ('No Aplica')",
      rpt.summary["columnas_con_nulos_semanticos"] > 0)
check("Detecta fechas como string (3)",      rpt.summary["fechas_como_string"] == 3,
      f"detectadas: {rpt.summary['fechas_como_string']}")

# PII — en este dataset hay teléfonos
print()
print("  Columnas con PII detectado:")
for col in rpt.pii_columns:
    cp = next(c for c in rpt.columns if c.name == col)
    print(f"    - {col}  [{cp.pii_type}]")

check("Detecta al menos 1 columna PII",   len(rpt.pii_columns) >= 1)

# Flags a nivel tabla
print()
print("  Flags de tabla:")
for f in rpt.quality_flags:
    print(f"    - {f}")

# Columnas con flags
print()
print("  Columnas con flags de calidad:")
for cp in rpt.columns:
    if cp.quality_flags:
        print(f"    {cp.name:<50} {cp.quality_flags}")

# ─────────────────────────────────────────────────────────────────────────────
separador("TEST 2 — Tabla sintetica con PII: clientes.csv")

tabla_cli = load_csv(ROOT / "data" / "catalogo" / "clientes.csv")
rpt_cli = profile(tabla_cli, table_name="clientes")

print(f"\n  {rpt_cli}")
print()
print("  Columnas con PII detectado:")
for col in rpt_cli.pii_columns:
    cp = next(c for c in rpt_cli.columns if c.name == col)
    print(f"    - {col}  [{cp.pii_type}]  muestra: {cp.sample_values[:2]}")

expected_pii = {"rut", "email", "telefono", "fecha_nacimiento", "nombre"}
found_pii = {
    next(c for c in rpt_cli.columns if c.name == col).pii_type
    for col in rpt_cli.pii_columns
}

check("Detecta rut como PII",             "rut" in found_pii)
check("Detecta email como PII",           "email" in found_pii)
check("Detecta telefono como PII",        "telefono" in found_pii)
check("Detecta nombre como PII",          "nombre" in found_pii)
check("Detecta fecha_nacimiento como PII","fecha_nacimiento" in found_pii)

# ─────────────────────────────────────────────────────────────────────────────
separador("TEST 3 — Tabla sintetica con salario: empleados.csv")

tabla_emp = load_csv(ROOT / "data" / "catalogo" / "empleados.csv")
rpt_emp = profile(tabla_emp, table_name="empleados")

pii_types_emp = {
    next(c for c in rpt_emp.columns if c.name == col).pii_type
    for col in rpt_emp.pii_columns
}
print(f"\n  PII detectado en empleados: {pii_types_emp}")
check("Detecta salario como PII",  "salario" in pii_types_emp)

# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
if errors == 0:
    print("  TODOS LOS CHECKS PASARON")
    print("  El perfilador esta listo para integrarse al agente (Hito 4).")
else:
    print(f"  {errors} CHECK(S) FALLARON")
    sys.exit(1)
print("=" * 65)
print()

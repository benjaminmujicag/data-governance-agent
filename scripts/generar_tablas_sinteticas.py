"""
Genera tablas sintéticas limpias para el catálogo interno de la empresa
ficticia "RetailBank S.A." — usadas en el Hito 1 como contrapunto al
dataset sucio real (DEIS/MINSAL).

Propósito didáctico:
- Proveer datos "ideales" para el catálogo base.
- Incluir columnas con PII (RUT, email, teléfono) para que el
  perfilador del Hito 2 practique la detección de datos sensibles.
- Contrastar "datos limpios" vs "datos reales sucios".

Tablas generadas en data/catalogo/:
    clientes.csv       — personas con RUT, email, teléfono (PII rico)
    transacciones.csv  — movimientos de cuenta/tarjeta
    productos.csv      — catálogo de productos retail
    empleados.csv      — personal interno (PII sensible)
    sucursales.csv     — red de sucursales con coordenadas

Ejecución:
    python scripts/generar_tablas_sinteticas.py
"""

import random
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CATALOGO = ROOT / "data" / "catalogo"
CATALOGO.mkdir(parents=True, exist_ok=True)

random.seed(42)  # reproducible

# ── utilidades ────────────────────────────────────────────────────────────────

def rut_chileno(n: int) -> str:
    """Genera un RUT chileno sintético (solo para datos de prueba)."""
    dv_chars = "0123456789K"
    return f"{n}-{random.choice(dv_chars)}"


def fecha_aleatoria(inicio: date, fin: date) -> str:
    delta = (fin - inicio).days
    return (inicio + timedelta(days=random.randint(0, delta))).isoformat()


# ── 1. clientes.csv ───────────────────────────────────────────────────────────

nombres = [
    "Ana García", "Carlos López", "María Rodríguez", "Juan Martínez",
    "Laura Soto", "Pedro Fuentes", "Claudia Herrera", "Diego Muñoz",
    "Valentina Torres", "Rodrigo Vargas", "Francisca Reyes", "Matías Silva",
    "Camila Morales", "Sebastián Castro", "Daniela Rojas", "Ignacio Pérez",
    "Javiera Núñez", "Felipe Vega", "Sofía Araya", "Tomás Espinoza",
]

dominios = ["gmail.com", "hotmail.com", "yahoo.com", "empresa.cl", "outlook.com"]
comunas = ["Santiago", "Providencia", "Las Condes", "Ñuñoa", "Maipú",
           "Pudahuel", "La Florida", "Vitacura", "San Miguel", "Conchalí"]

clientes = []
for i in range(1, 201):
    nombre = random.choice(nombres)
    partes = nombre.lower().replace("é", "e").replace("á", "a").replace("ó", "o").split()
    email = f"{partes[0]}.{partes[-1]}{random.randint(1, 99)}@{random.choice(dominios)}"
    clientes.append({
        "cliente_id":    i,
        "nombre":        nombre,
        "rut":           rut_chileno(10_000_000 + i * 37),
        "email":         email,
        "telefono":      f"+569{random.randint(10_000_000, 99_999_999)}",
        "fecha_nacimiento": fecha_aleatoria(date(1960, 1, 1), date(2000, 12, 31)),
        "comuna":        random.choice(comunas),
        "segmento":      random.choice(["Premium", "Standard", "Basic"]),
        "activo":        random.choice([True, True, True, False]),
    })

df_clientes = pd.DataFrame(clientes)
df_clientes.to_csv(CATALOGO / "clientes.csv", index=False, encoding="utf-8")
print(f"  clientes.csv          {len(df_clientes):>4} filas x {len(df_clientes.columns)} cols")

# ── 2. transacciones.csv ──────────────────────────────────────────────────────

categorias = ["Supermercado", "Farmacia", "Combustible", "Restaurante",
              "Ropa", "Electrónica", "Viaje", "Entretenimiento"]
tipos_tx   = ["Débito", "Crédito", "Transferencia"]

transacciones = []
for i in range(1, 501):
    transacciones.append({
        "tx_id":        f"TX{i:05d}",
        "cliente_id":   random.randint(1, 200),
        "fecha":        fecha_aleatoria(date(2024, 1, 1), date(2026, 6, 1)),
        "monto_clp":    round(random.uniform(500, 250_000), 0),
        "tipo":         random.choice(tipos_tx),
        "categoria":    random.choice(categorias),
        "comercio":     f"Comercio_{random.randint(1, 50):03d}",
        "aprobada":     random.choice([True, True, True, True, False]),
    })

df_tx = pd.DataFrame(transacciones)
df_tx.to_csv(CATALOGO / "transacciones.csv", index=False, encoding="utf-8")
print(f"  transacciones.csv     {len(df_tx):>4} filas x {len(df_tx.columns)} cols")

# ── 3. productos.csv ──────────────────────────────────────────────────────────

categorias_prod = {
    "Electrónica":  ["Smartphone", "Laptop", "Tablet", "Auriculares", "Smartwatch"],
    "Ropa":         ["Polera", "Pantalón", "Chaqueta", "Zapatillas", "Calcetines"],
    "Alimentos":    ["Yogurt", "Pan integral", "Jugo natural", "Café molido", "Granola"],
    "Farmacia":     ["Paracetamol", "Ibuprofeno", "Vitamina C", "Protector solar", "Mascarilla"],
}

productos = []
pid = 1
for cat, items in categorias_prod.items():
    for nombre in items:
        for variante in range(1, 4):
            productos.append({
                "producto_id":  f"P{pid:04d}",
                "nombre":       f"{nombre} v{variante}",
                "categoria":    cat,
                "precio_clp":   round(random.uniform(990, 499_990), 0),
                "stock":        random.randint(0, 500),
                "activo":       random.choice([True, True, True, False]),
                "proveedor_id": f"PROV{random.randint(1, 20):03d}",
            })
            pid += 1

df_prod = pd.DataFrame(productos)
df_prod.to_csv(CATALOGO / "productos.csv", index=False, encoding="utf-8")
print(f"  productos.csv         {len(df_prod):>4} filas x {len(df_prod.columns)} cols")

# ── 4. empleados.csv ──────────────────────────────────────────────────────────

cargos = ["Analista de Datos", "Gerente de Sucursal", "Ejecutivo de Cuenta",
          "Ingeniero de Software", "Jefa de Operaciones", "Data Engineer",
          "Compliance Officer", "Auditor Interno"]
deptos = ["Tecnología", "Operaciones", "Comercial", "Riesgos", "Cumplimiento"]

empleados = []
for i in range(1, 51):
    nombre = random.choice(nombres)
    partes = nombre.lower().replace("é", "e").replace("á", "a").replace("ó", "o").split()
    empleados.append({
        "empleado_id":   i,
        "nombre":        nombre,
        "rut":           rut_chileno(8_000_000 + i * 41),
        "email_corp":    f"{partes[0]}.{partes[-1]}@retailbank.cl",
        "cargo":         random.choice(cargos),
        "departamento":  random.choice(deptos),
        "salario_clp":   random.randint(800_000, 5_000_000),
        "fecha_ingreso": fecha_aleatoria(date(2015, 1, 1), date(2025, 12, 31)),
        "activo":        random.choice([True, True, True, False]),
    })

df_emp = pd.DataFrame(empleados)
df_emp.to_csv(CATALOGO / "empleados.csv", index=False, encoding="utf-8")
print(f"  empleados.csv         {len(df_emp):>4} filas x {len(df_emp.columns)} cols")

# ── 5. sucursales.csv ─────────────────────────────────────────────────────────

sucursales_data = [
    ("SUC001", "Sucursal Centro",       "Santiago",    -33.4569, -70.6483, "Av. Libertador 100"),
    ("SUC002", "Sucursal Las Condes",   "Las Condes",  -33.4136, -70.5809, "Av. Apoquindo 4500"),
    ("SUC003", "Sucursal Providencia",  "Providencia", -33.4317, -70.6062, "Av. Providencia 1200"),
    ("SUC004", "Sucursal Maipú",        "Maipú",       -33.5115, -70.7580, "Av. Américo Vespucio 300"),
    ("SUC005", "Sucursal La Florida",   "La Florida",  -33.5266, -70.5898, "Av. Vicuña Mackenna 7000"),
    ("SUC006", "Sucursal Valparaíso",   "Valparaíso",  -33.0472, -71.6127, "Av. Brasil 800"),
    ("SUC007", "Sucursal Viña del Mar", "Viña del Mar",-33.0245, -71.5518, "Av. San Martín 400"),
    ("SUC008", "Sucursal Concepción",   "Concepción",  -36.8270, -73.0495, "Av. O'Higgins 650"),
]

sucursales = []
for codigo, nombre, ciudad, lat, lon, direccion in sucursales_data:
    sucursales.append({
        "sucursal_id":   codigo,
        "nombre":        nombre,
        "ciudad":        ciudad,
        "direccion":     direccion,
        "latitud":       lat,
        "longitud":      lon,
        "telefono":      f"(+562) 2{random.randint(100, 999)}-{random.randint(1000, 9999)}",
        "horario":       "Lun-Vie 09:00-18:00",
        "activa":        True,
    })

df_suc = pd.DataFrame(sucursales)
df_suc.to_csv(CATALOGO / "sucursales.csv", index=False, encoding="utf-8")
print(f"  sucursales.csv        {len(df_suc):>4} filas x {len(df_suc.columns)} cols")

# ── Resumen ───────────────────────────────────────────────────────────────────

print()
print("Tablas generadas en data/catalogo/")
print()
print("Columnas con PII por tabla:")
pii_map = {
    "clientes":      ["rut", "email", "telefono", "fecha_nacimiento"],
    "transacciones": [],
    "productos":     [],
    "empleados":     ["rut", "email_corp", "salario_clp"],
    "sucursales":    ["telefono"],
}
for tabla, cols_pii in pii_map.items():
    if cols_pii:
        print(f"  {tabla:<15} PII: {', '.join(cols_pii)}")
    else:
        print(f"  {tabla:<15} (sin PII directo)")

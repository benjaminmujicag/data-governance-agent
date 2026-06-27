"""
Script de exploración del dataset público (Hito 1).

Objetivo didáctico: mostrar en concreto por qué los datos reales
son difíciles y qué tipos de suciedad tiene este dataset.
Esto motiva el perfilador automático del Hito 2.

Ejecución:
    python scripts/explorar_dataset.py
"""

import sys
from pathlib import Path

# Agregar la raíz del proyecto al path para que los imports funcionen
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.connectors.csv_loader import load_csv


def separador(titulo: str) -> None:
    print()
    print("=" * 60)
    print(f"  {titulo}")
    print("=" * 60)


def main() -> None:
    csv_path = ROOT / "data" / "publico" / "establecimientos_salud.csv"

    print("\n[EXPLORADOR] Explorando dataset: Establecimientos de Salud (DEIS/MINSAL)")
    print(f"   Archivo: {csv_path.name}")

    # --- Cargar con el conector ---
    tabla = load_csv(csv_path)

    separador("1. ¿QUÉ CARGAMOS?")
    print(f"  {tabla}")
    print(f"  Encoding detectado : {tabla.metadata['encoding']}")
    print(f"  Separador detectado: {tabla.metadata['separator']!r}")
    print(f"  Tamaño archivo     : {tabla.metadata['file_size_kb']} KB")
    if tabla.metadata["warnings"]:
        print("  [WARN] Advertencias al cargar:")
        for w in tabla.metadata["warnings"]:
            print(f"      - {w}")

    df = tabla.df

    separador("2. COLUMNAS Y TIPOS DE DATO")
    print(f"  {'Columna':<50} Tipo pandas")
    print(f"  {'-'*50} -----------")
    for col, dtype in df.dtypes.items():
        print(f"  {col:<50} {dtype}")

    separador("3. NULOS POR COLUMNA (solo las que tienen)")
    nulos = df.isnull().sum()
    pct = (nulos / len(df) * 100).round(1)
    tiene_nulos = nulos[nulos > 0].sort_values(ascending=False)

    print(f"  {'Columna':<50} {'Nulos':>7}  {'%':>6}")
    print(f"  {'-'*50} {'-------':>7}  {'------':>6}")
    for col in tiene_nulos.index:
        barra = "#" * int(pct[col] / 5)  # 1 # = 5%
        print(f"  {col:<50} {nulos[col]:>7,}  {pct[col]:>5.1f}%  {barra}")

    separador("4. PROBLEMAS ESPECÍFICOS DE CALIDAD")

    # 4a. Columnas de fecha guardadas como string
    date_cols = [c for c in df.columns if "fecha" in c.lower() or "date" in c.lower()]
    if date_cols:
        print(f"\n  a) Columnas de fecha guardadas como STRING (no datetime):")
        for c in date_cols:
            non_null = df[c].dropna()
            example = non_null.iloc[0] if len(non_null) > 0 else "—"
            print(f"     • {c}: dtype={df[c].dtype}, ejemplo={example!r}")

    # 4b. "No Aplica" como valor (semánticamente nulo)
    print(f"\n  b) Valores 'No Aplica' (semánticamente nulos, no NaN):")
    for col in df.select_dtypes(include=["object", "str"]).columns:
        count_na_text = (df[col] == "No Aplica").sum()
        if count_na_text > 0:
            pct_val = count_na_text / len(df) * 100
            print(f"     • {col}: {count_na_text:,} filas ({pct_val:.1f}%)")

    # 4c. Columna que mezcla dos campos
    mixed_cols = [c for c in df.columns if "_" in c and any(
        x in c.lower() for x in ["telefono", "codigo", "servicio"]
    )]
    if mixed_cols:
        print(f"\n  c) Columnas que parecen mezclar dos campos:")
        for c in mixed_cols:
            print(f"     • {c}")

    # 4d. Columna numérica guardada como string
    print(f"\n  d) Columna 'Numero' (dirección) guardada como string:")
    if "Numero" in df.columns:
        sample = df["Numero"].dropna().head(10).tolist()
        print(f"     dtype: {df['Numero'].dtype}")
        print(f"     Muestra de valores: {sample}")
        non_numeric = df["Numero"].dropna().apply(
            lambda x: not str(x).replace(".", "").replace("-", "").isdigit()
        ).sum()
        print(f"     Valores no numéricos: {non_numeric:,}")

    # 4e. Coordenadas con nulos
    if "Latitud" in df.columns:
        n_sin_geo = df["Latitud"].isnull().sum()
        print(f"\n  e) Establecimientos SIN geolocalización: {n_sin_geo:,} ({n_sin_geo/len(df)*100:.1f}%)")

    separador("5. MUESTRA DE FILAS (5 primeras, columnas clave)")
    cols_clave = [
        "EstablecimientoCodigo", "EstablecimientoGlosa",
        "RegionGlosa", "TipoEstablecimientoGlosa",
        "FechaInicioFuncionamientoEstab", "TelefonoMovil_TelefonoFijo",
        "Latitud",
    ]
    cols_existentes = [c for c in cols_clave if c in df.columns]
    print(df[cols_existentes].head(5).to_string(index=False))

    separador("RESUMEN EJECUTIVO")
    print("  Este dataset tiene:")
    total_nulos = df.isnull().sum().sum()
    total_celdas = df.shape[0] * df.shape[1]
    print(f"  • {df.shape[0]:,} registros y {df.shape[1]} columnas")
    print(f"  • {total_nulos:,} celdas nulas de {total_celdas:,} ({total_nulos/total_celdas*100:.1f}%)")
    print(f"  • Columnas con >50% nulos: {(pct > 50).sum()}")
    print(f"  • Fechas como strings (no parseadas): {len(date_cols)} columnas")
    print(f"  • Columnas con 'No Aplica' semántico: ", end="")
    no_aplica_cols = sum(
        1 for col in df.select_dtypes(include=["object", "str"]).columns
        if (df[col] == "No Aplica").sum() > 0
    )
    print(no_aplica_cols)
    print()
    print("  -> El Hito 2 (perfilador) automatizara TODO este diagnostico.")
    print()


if __name__ == "__main__":
    main()

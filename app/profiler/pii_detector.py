"""
Detector heurístico de PII (Personally Identifiable Information).

Estrategia en dos capas:
  1. Nombre de la columna  — palabras clave en el nombre sugieren PII.
  2. Muestra de valores    — regex sobre los primeros valores no-nulos
                             para confirmar el patrón.

Por qué heurístico y no ML:
  - Es explicable: puedes decir exactamente por qué marcó una columna.
  - No requiere datos de entrenamiento ni un modelo cargado.
  - Suficiente para el 90% de los casos de gobernanza real.
  - Igual a la capa 1 de herramientas como Microsoft Presidio y AWS Macie.

Tipos de PII que detectamos:
  rut             — identificador chileno  (ej: "12345678-9", "12.345.678-K")
  email           — correo electrónico
  telefono        — número de teléfono (fijo, celular, internacional)
  nombre          — nombre de persona
  fecha_nacimiento — fecha de nacimiento
  salario         — salario / remuneración
  direccion       — dirección postal
"""

import re
from typing import Optional

import pandas as pd

# ── 1. Palabras clave en el nombre de la columna ──────────────────────────────

# Mapeamos: palabra_clave → tipo_pii
# La búsqueda es case-insensitive y busca la palabra como substring.
_NAME_KEYWORDS: list[tuple[str, str]] = [
    # RUT
    ("rut",              "rut"),
    ("run",              "rut"),
    # Email
    ("email",            "email"),
    ("correo",           "email"),
    ("mail",             "email"),
    # Teléfono
    ("telefono",         "telefono"),
    ("fono",             "telefono"),
    ("phone",            "telefono"),
    ("celular",          "telefono"),
    ("movil",            "telefono"),
    # Nombre de persona
    ("nombre",           "nombre"),
    ("name",             "nombre"),
    ("apellido",         "nombre"),
    ("lastname",         "nombre"),
    # Fecha de nacimiento
    ("nacimiento",       "fecha_nacimiento"),
    ("birthday",         "fecha_nacimiento"),
    ("birth",            "fecha_nacimiento"),
    ("fecha_nac",        "fecha_nacimiento"),
    # Salario
    ("salario",          "salario"),
    ("sueldo",           "salario"),
    ("remuneracion",     "salario"),
    ("salary",           "salario"),
    ("wage",             "salario"),
    # Dirección
    ("direccion",        "direccion"),
    ("domicilio",        "direccion"),
    ("address",          "direccion"),
]

# ── 2. Expresiones regulares para validar valores ─────────────────────────────

_VALUE_PATTERNS: dict[str, re.Pattern] = {
    "rut": re.compile(
        r"^\d{1,2}\.?\d{3}\.?\d{3}-[\dkK]$"
    ),
    "email": re.compile(
        r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
    ),
    # Teléfono: dígitos con separadores opcionales, sin punto decimal (≠ float),
    # y descartando lo que parezca fecha (dd-mm-aaaa / aaaa-mm-dd).
    "telefono": re.compile(
        r"^[\+\(]?[\d\s\(\)\-]{7,20}$"
    ),
}

# Patrón de fecha: lo excluimos del chequeo de teléfono
_DATE_LIKE = re.compile(
    r"^\d{1,4}[-/\.]\d{1,2}[-/\.]\d{2,4}$"
)

# ── Función principal ─────────────────────────────────────────────────────────

def detect_pii(col_name: str, sample: pd.Series) -> tuple[bool, Optional[str]]:
    """
    Decide si una columna contiene PII y de qué tipo.

    Args:
        col_name:  Nombre de la columna.
        sample:    Serie de pandas con valores no-nulos de muestra.

    Returns:
        (pii_flag, pii_type) — si pii_flag es False, pii_type es None.

    Lógica:
        - Capa 1: si el nombre de columna contiene una palabra clave → PII.
        - Capa 2: si la capa 1 no disparó, revisa los valores con regex.
        - Si ninguna capa detecta nada → no es PII.
    """
    col_lower = col_name.lower().replace(" ", "_").replace("-", "_")

    # Capa 1: nombre de columna
    for keyword, pii_type in _NAME_KEYWORDS:
        if keyword in col_lower:
            return True, pii_type

    # Capa 2: regex sobre valores de muestra (solo si hay valores)
    if len(sample) == 0:
        return False, None

    sample_str = sample.dropna().astype(str).head(20)
    if len(sample_str) == 0:
        return False, None

    for pii_type, pattern in _VALUE_PATTERNS.items():
        # Para teléfono: excluir previamente los valores que parecen fechas
        # (18-12-2012, 2024-03-23, etc.) para evitar falsos positivos.
        if pii_type == "telefono":
            candidate = sample_str[
                ~sample_str.apply(lambda v: bool(_DATE_LIKE.match(v.strip())))
            ]
        else:
            candidate = sample_str

        if len(candidate) == 0:
            continue

        # Necesitamos que al menos el 50% de la muestra coincida
        # para evitar falsos positivos por coincidencias accidentales.
        matches = candidate.apply(lambda v: bool(pattern.match(v.strip())))
        match_rate = matches.mean()
        if match_rate >= 0.5:
            return True, pii_type

    return False, None

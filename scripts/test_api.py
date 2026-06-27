"""
test_api.py — Prueba la API por HTTP (Hito 5), actuando como CLIENTE.

Concepto: cliente vs servidor.
  El servidor (uvicorn corriendo app.api.main:app) escucha en un puerto. Este script es un
  CLIENTE que le manda peticiones HTTP con httpx, igual que lo haría un navegador, Streamlit
  u otro sistema. Por eso PRIMERO hay que levantar el servidor en otra terminal.

Qué verifica:
  - GET  /health   responde ok.
  - GET  /catalog  lista las tablas disponibles.
  - POST /profile  perfila 'clientes' y reporta PII (no usa LLM → no gasta cuota).
  - POST /ask      el agente responde una pregunta (SÍ usa LLM → gasta cuota).

Precondición — levantar el servidor en otra terminal:
  cd proyectos/asistente-rag-gobernanza
  .venv\\Scripts\\activate
  uvicorn app.api.main:app --reload
  (También: Docker abierto + corpus indexado para que /ask con search_governance funcione.)

Cómo correr (en una segunda terminal):
  .venv\\Scripts\\python.exe scripts/test_api.py
  Para saltarte la llamada al LLM (ahorrar cuota): agrega --no-ask
"""

import sys

import httpx

BASE_URL = "http://localhost:8000"
TIMEOUT = 120.0  # el agente puede tardar (varias llamadas al LLM + reintentos)


def main() -> None:
    skip_ask = "--no-ask" in sys.argv

    print("=" * 70)
    print(f"TEST DE LA API — cliente HTTP contra {BASE_URL}")
    print("=" * 70)

    passed = 0
    failed = 0

    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as client:
        # 1) Health: ¿está viva la API?
        try:
            r = client.get("/health")
            assert r.status_code == 200, f"status {r.status_code}"
            assert r.json().get("status") == "ok"
            print("[OK] GET /health ->", r.json())
            passed += 1
        except Exception as exc:  # noqa: BLE001
            print("[FALLO] GET /health ->", repr(exc))
            print("  ¿Levantaste el servidor? uvicorn app.api.main:app --reload")
            failed += 1
            _summary(passed, failed)
            sys.exit(1)  # sin servidor, el resto no tiene sentido

        # 2) Catalog: lista de tablas
        try:
            r = client.get("/catalog")
            assert r.status_code == 200, f"status {r.status_code}"
            disponibles = r.json().get("tablas_disponibles", [])
            assert "clientes" in disponibles
            print("[OK] GET /catalog -> disponibles:", disponibles)
            passed += 1
        except Exception as exc:  # noqa: BLE001
            print("[FALLO] GET /catalog ->", repr(exc))
            failed += 1

        # 3) Profile: perfila 'clientes' (sin LLM)
        try:
            r = client.post("/profile", json={"table_name": "clientes"})
            assert r.status_code == 200, f"status {r.status_code}: {r.text}"
            data = r.json()
            pii = data.get("columnas_pii", [])
            assert "rut" in pii, f"esperaba 'rut' en PII, vino {pii}"
            print(f"[OK] POST /profile -> shape={data.get('shape')} pii={pii}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            print("[FALLO] POST /profile ->", repr(exc))
            failed += 1

        # 4) Ask: el agente responde (usa LLM)
        if skip_ask:
            print("[SKIP] POST /ask (--no-ask)")
        else:
            try:
                r = client.post(
                    "/ask",
                    json={"question": "¿Qué columnas tienen PII en la tabla de clientes?"},
                )
                assert r.status_code == 200, f"status {r.status_code}: {r.text}"
                data = r.json()
                assert len(data.get("tools_used", [])) > 0, "el agente no usó ninguna tool"
                print(f"[OK] POST /ask -> tools={[t['name'] for t in data['tools_used']]}")
                print("  RESPUESTA:", data["answer"][:300], "...")
                passed += 1
            except Exception as exc:  # noqa: BLE001
                print("[FALLO] POST /ask ->", repr(exc))
                failed += 1

    _summary(passed, failed)
    sys.exit(0 if failed == 0 else 1)


def _summary(passed: int, failed: int) -> None:
    print("=" * 70)
    print(f"RESULTADO: {passed} OK, {failed} fallo(s)")
    if failed == 0:
        print("La API funciona. Próximo paso: Hito 6 — UI con Streamlit.")
    print("=" * 70)


if __name__ == "__main__":
    main()

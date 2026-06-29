# Dockerfile — receta para construir la IMAGEN de la app (FastAPI + Streamlit).
#
# Imagen vs contenedor (la analogia clave):
#   - IMAGEN  = la receta / molde congelado: app + dependencias + como arrancar. Inmutable.
#   - CONTENEDOR = un plato cocinado a partir de la receta: una ejecucion viva de la imagen.
#   De una imagen puedes levantar muchos contenedores iguales. Por eso "corre igual en todos lados".
#
# Cada instruccion (FROM, COPY, RUN...) crea una CAPA. Docker cachea las capas: si una capa y
# todo lo anterior no cambian, la reutiliza. Por eso el orden importa (ver el truco de requirements).

# 1) Base: un Python 3.12 minimo ('slim' = sin paquetes de sistema innecesarios = imagen mas chica).
FROM python:3.12-slim

# 2) Variables de entorno utiles para Python dentro de contenedores:
#    - PYTHONDONTWRITEBYTECODE: no generar archivos .pyc (no sirven en una imagen efimera).
#    - PYTHONUNBUFFERED: imprimir logs al instante (sin buffer) para verlos en `docker logs`.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3) Carpeta de trabajo dentro de la imagen. Todo lo que sigue ocurre en /app.
WORKDIR /app

# 4) TRUCO DE CACHE: copiar PRIMERO solo requirements.txt e instalar.
#    Asi, si cambias codigo pero NO las dependencias, Docker reutiliza esta capa cara
#    (la instalacion de pip) y el rebuild es casi instantaneo. Si copiaramos todo el codigo
#    antes de instalar, cualquier cambio de codigo invalidaria el cache y re-instalaria todo.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5) Ahora si, el resto del codigo y los datos del proyecto.
#    (.dockerignore evita copiar .venv, pgdata, .env, etc.)
COPY . .

# 6) Documenta que la API escucha en el 8000 (informativo; el mapeo real esta en compose).
EXPOSE 8000

# 7) Comando por defecto: levantar la API con uvicorn.
#    OJO --host 0.0.0.0: dentro del contenedor hay que escuchar en TODAS las interfaces.
#    Si usaramos 127.0.0.1 (localhost), solo seria accesible DENTRO del contenedor y el
#    mapeo de puertos de Docker no podria alcanzarlo desde tu maquina. Error clasico.
#
#    PUERTO DINAMICO ($PORT): Cloud Run NO usa un puerto fijo; inyecta la variable de
#    entorno PORT (normalmente 8080) y espera que el contenedor escuche ahi. En local
#    (Docker Compose) PORT no esta definida, asi que ${PORT:-8000} usa 8000 por defecto.
#    Usamos la forma "sh -c ... exec" para que: (a) el shell expanda $PORT, y (b) exec
#    reemplace al shell por uvicorn (asi uvicorn recibe las senales SIGTERM de Cloud Run).
#    La UI (Streamlit) reutiliza esta MISMA imagen con otro comando (ver docker-compose.yml).
CMD ["sh", "-c", "exec uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

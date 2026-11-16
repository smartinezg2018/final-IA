# RAG Python Docs

Asistente que responde preguntas sobre programación en Python usando como fuente una biblioteca de libros en PDF.

## Qué está implementado

- **`procesar_pdfs.py`**: extrae el texto de los PDF de la carpeta `pdfs_python/` con PyMuPDF, lo limpia y lo divide en chunks (por defecto 1000 caracteres con 150 de solapamiento). Cada chunk guarda su libro y página de origen. El resultado se guarda en `chunks.json`.
- **`indexar_chunks.py`**: convierte cada chunk en un vector con el modelo de embeddings `intfloat/multilingual-e5-small` y lo guarda en una base vectorial de Chroma (`chroma_db/`).
- **`rag.py`**: núcleo compartido; carga el modelo de embeddings y la colección de Chroma, y busca los chunks más parecidos a una pregunta.
- **`api.py`**: API en FastAPI que recupera los chunks más relevantes y se los pasa a Gemini para generar una respuesta con citas.
  - `GET /` — interfaz web de chat (`static/index.html`).
  - `GET /api/estado` — chunks indexados y si Gemini está configurado.
  - `GET /api/libros` — lista de PDF disponibles.
  - `POST /api/buscar` — solo búsqueda de chunks, sin Gemini.
  - `POST /api/preguntar` — búsqueda + respuesta generada por Gemini.
  - `/pdfs/<archivo>` — sirve los PDF para abrir cada referencia en su página.

## Instalación

Requisitos: Python 3.10 o superior y [uv](https://docs.astral.sh/uv/).

Si no tienes uv, instálalo en PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Crea el entorno virtual e instala las dependencias:

```powershell
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

Crea el archivo `.env` a partir del ejemplo y pon tu API key de Gemini:

```powershell
copy .env.example .env
notepad .env
```

## Preparar los datos

Con el entorno activado, ejecuta los dos pasos en orden.

**1. Crear los chunks** (lee `pdfs_python/` y genera `chunks.json`):

```powershell
python procesar_pdfs.py
```

**2. Vectorizar los chunks** (lee `chunks.json` y crea la base vectorial en `chroma_db/`):

```powershell
python indexar_chunks.py
```

La primera vez descarga el modelo de embeddings. Si el índice ya está creado, se omite la indexación.

## Ejecutar la aplicación

```powershell
uvicorn api:app --reload
```

Abre <http://127.0.0.1:8000> para usar el chat, o <http://127.0.0.1:8000/docs> para probar la API desde Swagger.

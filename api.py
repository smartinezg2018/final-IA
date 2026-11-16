"""API RAG: busca contexto en Chroma y genera la respuesta con Gemini.

Ejecutar con:
    uvicorn api:app --reload
y abrir http://127.0.0.1:8000
"""

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from rag import buscar, coleccion

BASE = Path(__file__).parent
CARPETA_PDFS = BASE / "pdfs_python"

# La API key va en .env
load_dotenv(BASE / ".env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODELO = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

cliente_gemini = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

INSTRUCCIONES = """Eres un asistente que responde preguntas sobre programación en Python.
Responde únicamente con la información de los fragmentos de contexto que se te entregan.
Cita los fragmentos que uses con su número entre corchetes, por ejemplo [1] o [2][3].
Si el contexto no contiene la respuesta, dilo claramente en lugar de inventarla.
Responde en el mismo idioma de la pregunta y usa Markdown (bloques de código cuando corresponda)."""

app = FastAPI(title="RAG Python Docs")

# Sirve los PDFs para que el front pueda abrir cada referencia en su página (/pdfs/<archivo>#page=N).
if CARPETA_PDFS.is_dir():
    app.mount("/pdfs", StaticFiles(directory=CARPETA_PDFS), name="pdfs")


class Turno(BaseModel):
    rol: Literal["usuario", "asistente"]
    texto: str = Field(max_length=8000)


class Consulta(BaseModel):
    pregunta: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)
    # Turnos anteriores del chat (opcional). Solo se usan para que Gemini entienda el contexto
    # de la conversación; la búsqueda en Chroma se hace con la pregunta actual.
    historial: list[Turno] = Field(default_factory=list, max_length=20)


def construir_prompt(pregunta, chunks):
    contexto = "\n\n".join(
        f"[{i}] (fuente: {c['metadata']['source']}, pág. {c['metadata']['page']})\n{c['texto']}"
        for i, c in enumerate(chunks, start=1)
    )
    return f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"


def construir_contenidos(consulta, chunks):
    """Historial del chat + la pregunta actual con su contexto, en el formato de Gemini."""
    contenidos = [
        types.Content(role="user" if t.rol == "usuario" else "model", parts=[types.Part(text=t.texto)])
        for t in consulta.historial
        if t.texto.strip()
    ]
    contenidos.append(
        types.Content(role="user", parts=[types.Part(text=construir_prompt(consulta.pregunta, chunks))])
    )
    return contenidos


@app.get("/")
def inicio():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/api/estado")
def estado():
    return {
        "chunks_indexados": coleccion.count(),
        "gemini_configurado": cliente_gemini is not None,
        "modelo_gemini": GEMINI_MODELO,
    }


@app.get("/api/libros")
def libros():
    """Lista los PDFs de la biblioteca (para mostrarlos en el front)."""
    if not CARPETA_PDFS.is_dir():
        return {"libros": []}
    return {"libros": sorted(p.name for p in CARPETA_PDFS.glob("*.pdf"))}


@app.post("/api/buscar")
def endpoint_buscar(consulta: Consulta):
    """Solo recuperación: devuelve los top-k chunks sin llamar a Gemini."""
    return {"fuentes": buscar(consulta.pregunta, consulta.k)}


@app.post("/api/preguntar")
def endpoint_preguntar(consulta: Consulta):
    """Recupera los top-k chunks y se los pasa a Gemini como contexto."""
    if cliente_gemini is None:
        raise HTTPException(500, "Falta GEMINI_API_KEY en .env")

    chunks = buscar(consulta.pregunta, consulta.k)
    try:
        respuesta = cliente_gemini.models.generate_content(
            model=GEMINI_MODELO,
            contents=construir_contenidos(consulta, chunks),
            config=types.GenerateContentConfig(system_instruction=INSTRUCCIONES, temperature=0.2),
        )
    except Exception as error:
        raise HTTPException(502, f"Error al llamar a Gemini: {error}") from error

    return {"respuesta": respuesta.text, "fuentes": chunks, "modelo": GEMINI_MODELO}

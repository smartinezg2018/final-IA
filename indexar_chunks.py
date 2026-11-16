"""Crea un índice vectorial con los chunks de chunks.json y permite buscar en él."""

import json

import chromadb
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODELO = "intfloat/multilingual-e5-small"
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"

modelo = SentenceTransformer(MODELO, device=DISPOSITIVO)
if DISPOSITIVO == "cuda":
    modelo.half()  # float16: más rápido en GPU, casi sin pérdida de calidad
coleccion = chromadb.PersistentClient("chroma_db").get_or_create_collection(
    "python_docs", metadata={"hnsw:space": "cosine"}
)


def cargar_chunks(ruta="chunks.json"):
    with open(ruta, encoding="utf-8") as archivo:
        return json.load(archivo)


def crear_indice(chunks, lote=500):
    """Convierte cada chunk en un vector y lo guarda en Chroma."""
    with tqdm(total=len(chunks), desc="Indexando", unit="chunk") as barra:
        for i in range(0, len(chunks), lote):
            grupo = chunks[i : i + lote]
            textos = [chunk["texto"] for chunk in grupo]
            vectores = modelo.encode(
                ["passage: " + t for t in textos], batch_size=128, normalize_embeddings=True
            )
            coleccion.upsert(
                ids=[chunk["chunk_id"] for chunk in grupo],
                documents=textos,
                embeddings=vectores.tolist(),
                metadatas=[chunk["metadata"] for chunk in grupo],
            )
            barra.update(len(grupo))


def buscar(pregunta, k=3):
    """Devuelve los k chunks más parecidos a la pregunta."""
    vector = modelo.encode("query: " + pregunta, normalize_embeddings=True)
    resultado = coleccion.query(query_embeddings=[vector.tolist()], n_results=k)
    return list(zip(resultado["documents"][0], resultado["metadatas"][0]))


def main():
    chunks = cargar_chunks()
    if coleccion.count() < len(chunks):
        crear_indice(chunks)
    else:
        print("Índice ya creado, se omite la indexación.")
    print(f"Chunks: {len(chunks)} | Indexados: {coleccion.count()}")

    for pregunta in ["¿Cómo funciona un decorador en Python?"]:
        print(f"\nPregunta: {pregunta}")
        for texto, meta in buscar(pregunta):
            print(f"- {meta['source']} (pág. {meta['page']}): {texto[:150]!r}")


if __name__ == "__main__":
    main()

"""Crea un índice vectorial con los chunks de chunks.json y permite buscar en él."""

import json

import chromadb
from sentence_transformers import SentenceTransformer

MODELO = "intfloat/multilingual-e5-small"

modelo = SentenceTransformer(MODELO)
coleccion = chromadb.PersistentClient("chroma_db").get_or_create_collection(
    "python_docs", metadata={"hnsw:space": "cosine"}
)


def cargar_chunks(ruta="chunks.json"):
    with open(ruta, encoding="utf-8") as archivo:
        return json.load(archivo)


def crear_indice(chunks, lote=500):
    """Convierte cada chunk en un vector y lo guarda en Chroma."""
    for i in range(0, len(chunks), lote):
        grupo = chunks[i : i + lote]
        textos = [chunk["texto"] for chunk in grupo]
        vectores = modelo.encode(["passage: " + t for t in textos], normalize_embeddings=True)
        coleccion.upsert(
            ids=[chunk["chunk_id"] for chunk in grupo],
            documents=textos,
            embeddings=vectores.tolist(),
            metadatas=[chunk["metadata"] for chunk in grupo],
        )


def buscar(pregunta, k=3):
    """Devuelve los k chunks más parecidos a la pregunta."""
    vector = modelo.encode("query: " + pregunta, normalize_embeddings=True)
    resultado = coleccion.query(query_embeddings=[vector.tolist()], n_results=k)
    return list(zip(resultado["documents"][0], resultado["metadatas"][0]))


def main():
    chunks = cargar_chunks()
    crear_indice(chunks)
    print(f"Chunks: {len(chunks)} | Indexados: {coleccion.count()}")

    for pregunta in ["¿Cómo funciona un decorador en Python?"]:
        print(f"\nPregunta: {pregunta}")
        for texto, meta in buscar(pregunta):
            print(f"- {meta['source']} (pág. {meta['page']}): {texto[:150]!r}")


if __name__ == "__main__":
    main()

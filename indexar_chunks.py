"""Crea un índice vectorial con los chunks de chunks.json y permite buscar en él."""

import json

from tqdm import tqdm

from rag import buscar, coleccion, modelo


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


def main():
    print("Descargando los chunks")
    chunks = cargar_chunks()
    if coleccion.count() < len(chunks):
        crear_indice(chunks)
    else:
        print("Índice ya creado, se omite la indexación.")
    print(f"Chunks: {len(chunks)} | Indexados: {coleccion.count()}")

    print("Realizar preguntas")
    for pregunta in ["¿Cómo funciona un decorador en Python?"]:
        print(f"\nPregunta: {pregunta}")
        for chunk in buscar(pregunta):
            meta = chunk["metadata"]
            print(f"- {meta['source']} (pág. {meta['page']}): {chunk['texto'][:150]!r}")


if __name__ == "__main__":
    main()

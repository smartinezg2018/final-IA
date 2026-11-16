"""Núcleo RAG: modelo de embeddings, colección de Chroma y búsqueda de contexto.

Lo usan tanto `indexar_chunks.py` (para crear el índice) como `api.py` (para consultarlo).
"""

import chromadb
import torch
from sentence_transformers import SentenceTransformer

MODELO = "intfloat/multilingual-e5-small"
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"

modelo = SentenceTransformer(MODELO, device=DISPOSITIVO)
if DISPOSITIVO == "cuda":
    modelo.half()  # float16: más rápido en GPU, casi sin pérdida de calidad
coleccion = chromadb.PersistentClient("chroma_db").get_or_create_collection(
    "python_docs", metadata={"hnsw:space": "cosine"}
)


def buscar(pregunta, k=3):
    """Devuelve los k chunks más parecidos a la pregunta, con su similitud (0 a 1)."""
    vector = modelo.encode("query: " + pregunta, normalize_embeddings=True)
    resultado = coleccion.query(query_embeddings=[vector.tolist()], n_results=k)
    return [
        {"texto": texto, "metadata": meta, "similitud": round(1 - distancia, 4)}
        for texto, meta, distancia in zip(
            resultado["documents"][0], resultado["metadatas"][0], resultado["distances"][0]
        )
    ]

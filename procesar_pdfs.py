"""Extracción de texto de PDF y división en chunks para un sistema RAG.

Este módulo se limita a preparar el texto y sus metadatos. No crea embeddings,
no construye índices vectoriales y no utiliza modelos de lenguaje.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter


PaginaExtraida = dict[str, Any]
Chunk = dict[str, Any]
InformeExtraccion = dict[str, Any]


def _listar_pdfs(carpeta_pdfs: Path) -> list[Path]:
    """Devuelve los PDF directos de una carpeta en un orden estable."""
    return sorted(
        (
            ruta
            for ruta in carpeta_pdfs.iterdir()
            if ruta.is_file() and ruta.suffix.lower() == ".pdf"
        ),
        key=lambda ruta: ruta.name.casefold(),
    )


def extraer_texto_pdfs(
    carpeta_pdfs: str | Path,
) -> tuple[list[PaginaExtraida], InformeExtraccion]:
    """Extrae el texto de todas las páginas legibles de una carpeta de PDF.

    Las páginas se numeran desde 1 para que coincidan con la numeración que ve
    una persona al abrir el documento. Un error en un PDF o en una página no
    impide procesar los demás archivos.

    Args:
        carpeta_pdfs: Carpeta que contiene los documentos PDF.

    Returns:
        Una tupla con las páginas que tienen texto y un informe de extracción.

    Raises:
        FileNotFoundError: Si la carpeta indicada no existe.
        NotADirectoryError: Si la ruta indicada no es una carpeta.
    """
    carpeta = Path(carpeta_pdfs)
    if not carpeta.exists():
        raise FileNotFoundError(f"La carpeta de PDF no existe: {carpeta}")
    if not carpeta.is_dir():
        raise NotADirectoryError(f"La ruta no es una carpeta: {carpeta}")

    archivos_pdf = _listar_pdfs(carpeta)
    paginas_extraidas: list[PaginaExtraida] = []
    informe: InformeExtraccion = {
        "documentos_encontrados": len(archivos_pdf),
        "documentos_procesados": 0,
        "documentos_con_texto": 0,
        "paginas_leidas": 0,
        "paginas_con_texto": 0,
        "paginas_sin_texto": 0,
        "documentos_que_requieren_ocr": [],
        "errores": [],
    }

    for ruta_pdf in archivos_pdf:
        documento_tiene_texto = False
        paginas_leidas_documento = 0
        errores_paginas_documento = 0

        try:
            # Convertir a str mantiene compatibilidad con versiones de PyMuPDF
            # que no aceptan directamente objetos pathlib.Path.
            with fitz.open(str(ruta_pdf)) as documento:
                if documento.needs_pass:
                    raise ValueError("el PDF está protegido con contraseña")

                informe["documentos_procesados"] += 1

                for indice_pagina in range(documento.page_count):
                    numero_pagina = indice_pagina + 1
                    try:
                        pagina = documento.load_page(indice_pagina)
                        texto = pagina.get_text("text").strip()
                        paginas_leidas_documento += 1
                        informe["paginas_leidas"] += 1
                    except Exception as error:  # PyMuPDF puede lanzar varios tipos.
                        errores_paginas_documento += 1
                        informe["errores"].append(
                            {
                                "source": ruta_pdf.name,
                                "page": numero_pagina,
                                "error": f"No se pudo leer la página: {error}",
                            }
                        )
                        continue

                    if not texto:
                        informe["paginas_sin_texto"] += 1
                        continue

                    documento_tiene_texto = True
                    informe["paginas_con_texto"] += 1
                    paginas_extraidas.append(
                        {
                            "texto": texto,
                            "metadata": {
                                "source": ruta_pdf.name,
                                "page": numero_pagina,
                            },
                        }
                    )

                if documento_tiene_texto:
                    informe["documentos_con_texto"] += 1
                elif paginas_leidas_documento > 0 and errores_paginas_documento == 0:
                    # PyMuPDF no aplica OCR. Si ninguna página contiene texto,
                    # probablemente se trata de un PDF compuesto por imágenes.
                    informe["documentos_que_requieren_ocr"].append(ruta_pdf.name)
                elif documento.page_count == 0:
                    informe["errores"].append(
                        {
                            "source": ruta_pdf.name,
                            "page": None,
                            "error": "El PDF no contiene páginas.",
                        }
                    )

        except Exception as error:  # Permite continuar con el siguiente PDF.
            informe["errores"].append(
                {
                    "source": ruta_pdf.name,
                    "page": None,
                    "error": f"No se pudo abrir o procesar el PDF: {error}",
                }
            )

    return paginas_extraidas, informe


def _nombre_para_chunk(nombre_archivo: str) -> str:
    """Convierte el nombre del archivo en una parte segura y legible del ID."""
    nombre_sin_extension = Path(nombre_archivo).stem
    nombre_normalizado = re.sub(r"[^\w-]+", "_", nombre_sin_extension).strip("_")
    return nombre_normalizado or "documento"


def dividir_texto_en_chunks(
    paginas: list[PaginaExtraida],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    """Divide el texto extraído sin mezclar el contenido de distintas páginas.

    Args:
        paginas: Resultado de :func:`extraer_texto_pdfs`.
        chunk_size: Cantidad máxima aproximada de caracteres por chunk.
        chunk_overlap: Caracteres compartidos entre chunks consecutivos.

    Returns:
        Lista de chunks con identificador, texto y metadatos de procedencia.

    Raises:
        ValueError: Si la configuración de tamaño y solapamiento no es válida.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size debe ser mayor que 0.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap no puede ser negativo.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap debe ser menor que chunk_size.")

    separador = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks: list[Chunk] = []

    for pagina in paginas:
        texto_pagina = str(pagina.get("texto", "")).strip()
        metadata = pagina.get("metadata", {})
        source = str(metadata.get("source", "documento.pdf"))
        page = metadata.get("page")

        if not texto_pagina:
            continue

        fragmentos = separador.split_text(texto_pagina)
        numero_chunk = 0

        for fragmento in fragmentos:
            texto_chunk = fragmento.strip()
            if not texto_chunk:
                continue

            numero_chunk += 1
            chunks.append(
                {
                    "chunk_id": (
                        f"{_nombre_para_chunk(source)}_pagina{page}_chunk{numero_chunk}"
                    ),
                    "texto": texto_chunk,
                    "metadata": {
                        "source": source,
                        "page": page,
                        "chunk_number": numero_chunk,
                    },
                }
            )

    return chunks


def guardar_chunks_json(chunks: list[Chunk], ruta_salida: str | Path) -> Path:
    """Guarda los chunks como JSON UTF-8 y devuelve la ruta utilizada."""
    ruta = Path(ruta_salida)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    with ruta.open("w", encoding="utf-8") as archivo:
        json.dump(chunks, archivo, ensure_ascii=False, indent=2)

    return ruta


def verificar_chunks_json(ruta_json: str | Path, cantidad_esperada: int) -> list[Chunk]:
    """Vuelve a leer el JSON y valida su estructura básica y cantidad."""
    ruta = Path(ruta_json)
    with ruta.open("r", encoding="utf-8") as archivo:
        contenido = json.load(archivo)

    if not isinstance(contenido, list):
        raise ValueError("El JSON generado no contiene una lista.")
    if len(contenido) != cantidad_esperada:
        raise ValueError(
            "La cantidad de chunks del JSON no coincide con la cantidad generada."
        )

    for posicion, chunk in enumerate(contenido, start=1):
        if not isinstance(chunk, dict):
            raise ValueError(f"El chunk {posicion} no es un objeto JSON.")
        if not {"chunk_id", "texto", "metadata"}.issubset(chunk):
            raise ValueError(f"El chunk {posicion} no tiene la estructura requerida.")
        if not isinstance(chunk["metadata"], dict):
            raise ValueError(f"Los metadatos del chunk {posicion} no son un objeto.")
        if not {"source", "page", "chunk_number"}.issubset(chunk["metadata"]):
            raise ValueError(f"Los metadatos del chunk {posicion} están incompletos.")

    return contenido


def _crear_parser() -> argparse.ArgumentParser:
    """Configura los argumentos disponibles en la línea de comandos."""
    carpeta_proyecto = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Extrae texto de PDF y lo divide en chunks con metadatos."
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=carpeta_proyecto / "pdfs_python",
        help="Carpeta de PDF (predeterminado: pdfs_python).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=carpeta_proyecto / "chunks.json",
        help="Archivo JSON de salida (predeterminado: chunks.json).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Tamaño máximo aproximado del chunk (predeterminado: 500).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Solapamiento entre chunks (predeterminado: 100).",
    )
    return parser


def _imprimir_informe(informe: InformeExtraccion, cantidad_chunks: int) -> None:
    """Muestra un resumen legible de la extracción."""
    print("\nResumen del procesamiento")
    print("-------------------------")
    print(f"Documentos encontrados: {informe['documentos_encontrados']}")
    print(f"Documentos procesados:  {informe['documentos_procesados']}")
    print(f"Documentos con texto:   {informe['documentos_con_texto']}")
    print(f"Páginas leídas:          {informe['paginas_leidas']}")
    print(f"Páginas con texto:       {informe['paginas_con_texto']}")
    print(f"Páginas sin texto:       {informe['paginas_sin_texto']}")
    print(f"Chunks generados:        {cantidad_chunks}")

    documentos_ocr = informe["documentos_que_requieren_ocr"]
    if documentos_ocr:
        print("\nPDF sin texto extraíble; necesitan OCR (no implementado):")
        for nombre in documentos_ocr:
            print(f"  - {nombre}")

    if informe["errores"]:
        print("\nErrores encontrados:")
        for error in informe["errores"]:
            pagina = error["page"]
            ubicacion = f", página {pagina}" if pagina is not None else ""
            print(f"  - {error['source']}{ubicacion}: {error['error']}")


def main() -> int:
    """Ejecuta el flujo completo desde la línea de comandos."""
    argumentos = _crear_parser().parse_args()

    try:
        paginas, informe = extraer_texto_pdfs(argumentos.pdf_dir)
        if informe["documentos_encontrados"] == 0:
            print(f"No se encontraron archivos PDF en: {argumentos.pdf_dir}")
            return 1

        chunks = dividir_texto_en_chunks(
            paginas,
            chunk_size=argumentos.chunk_size,
            chunk_overlap=argumentos.chunk_overlap,
        )
        ruta_json = guardar_chunks_json(chunks, argumentos.output)
        chunks_verificados = verificar_chunks_json(ruta_json, len(chunks))
    except (FileNotFoundError, NotADirectoryError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    _imprimir_informe(informe, len(chunks))
    print(f"\nJSON guardado y verificado correctamente en: {ruta_json.resolve()}")

    print(f"\nEjemplos de chunks ({min(3, len(chunks_verificados))}):")
    for chunk in chunks_verificados[:3]:
        print(json.dumps(chunk, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

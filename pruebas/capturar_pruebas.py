"""Ejecuta los cinco casos de prueba sobre la interfaz web y guarda evidencia.

Para cada caso:
  - escribe la pregunta en el chat (modo Gemini o solo búsqueda),
  - guarda la respuesta completa de la API en `pruebas/resultados/casoN.json`,
  - toma una captura de la interfaz en `pruebas/capturas/casoN.png`.

Además renderiza la página del PDF original indicada en `PAGINAS_PDF` para comparar
la respuesta con el documento fuente (`pruebas/capturas/casoN_pdf_pXXX.png`).

Requisitos (además de requirements.txt):
    uv pip install playwright
Usa el Chrome instalado en el equipo (channel="chrome"), así que no hace falta
`playwright install`. La API debe estar corriendo:
    uvicorn api:app
    python pruebas/capturar_pruebas.py
"""

import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = Path(__file__).resolve().parent
RAIZ = BASE.parent
CAPTURAS = BASE / "capturas"
RESULTADOS = BASE / "resultados"
URL = "http://127.0.0.1:8000"
K = 5

# modo: "gemini" -> /api/preguntar (pasa por el LLM); "busqueda" -> /api/buscar (solo recuperación)
CASOS = [
    {"id": 1, "tipo": "habitual", "modo": "gemini",
     "pregunta": "¿Qué diferencia hay entre una lista y una tupla?"},
    {"id": 2, "tipo": "habitual", "modo": "gemini",
     "pregunta": "¿Cómo leo y escribo archivos de texto en Python?"},
    {"id": 3, "tipo": "habitual", "modo": "gemini",
     "pregunta": "¿Cómo manejo excepciones con try y except?"},
    {"id": 4, "tipo": "documentos", "modo": "busqueda",
     "pregunta": "¿Para qué sirve el módulo pyperclip y cómo copio texto al portapapeles?"},
    {"id": 5, "tipo": "límite", "modo": "gemini",
     "pregunta": "¿Cómo creo un DataFrame en pandas y filtro sus filas?"},
]

# Páginas del PDF original que se renderizan para comparar con la respuesta.
# En el caso 2 se incluye la pág. 231, que NO se recuperó, para evidenciar la limitación.
PAGINAS_PDF = {
    1: [("Python Crash Course", 102)],
    2: [("Python Crash Course", 230), ("Python Crash Course", 231)],
    3: [("Learning Python", 1148)],
    4: [("Automate the Boring Stuff", 159)],
}


def renderizar_pdf(caso_id, prefijo_libro, pagina):
    try:
        import pymupdf  # PyMuPDF, ya está en requirements.txt
    except ImportError:
        print("  (PyMuPDF no instalado: se omite la captura del PDF)")
        return
    pdf = next((p for p in (RAIZ / "pdfs_python").glob("*.pdf") if p.name.startswith(prefijo_libro)), None)
    if pdf is None:
        print(f"  (no se encontró el PDF '{prefijo_libro}')")
        return
    with pymupdf.open(pdf) as doc:
        pix = doc[pagina - 1].get_pixmap(dpi=110)
        destino = CAPTURAS / f"caso{caso_id}_pdf_p{pagina}.png"
        pix.save(destino)
        print(f"  PDF original -> {destino.relative_to(RAIZ)}")


def ejecutar_caso(pagina, caso):
    print(f"Caso {caso['id']} ({caso['tipo']}, {caso['modo']}): {caso['pregunta']}")
    pagina.click("#nueva")
    pagina.select_option("#modo", caso["modo"])
    pagina.select_option("#k", str(K))
    pagina.fill("#pregunta", caso["pregunta"])

    ruta = "/api/preguntar" if caso["modo"] == "gemini" else "/api/buscar"
    with pagina.expect_response(lambda r: r.url.endswith(ruta), timeout=120_000) as info:
        pagina.click("#enviar")
    respuesta = info.value
    datos = respuesta.json()
    pagina.wait_for_selector(".msg.asistente[data-id]", timeout=30_000)
    pagina.wait_for_timeout(800)  # deja terminar el render y el resaltado de código

    # Abre el primer fragmento en el panel de referencias para que se vea completo.
    if pagina.locator(".frag").count():
        pagina.locator(".frag .ver-mas").first.click()
    pagina.evaluate("document.querySelector('#scroll').scrollTop = 0")

    registro = {**caso, "status": respuesta.status, "respuesta_api": datos}
    (RESULTADOS / f"caso{caso['id']}.json").write_text(
        json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")

    # Ajusta el alto del viewport al contenido del chat para capturar la respuesta completa.
    alto = pagina.evaluate("document.querySelector('#mensajes').scrollHeight") + 260
    pagina.set_viewport_size({"width": 1500, "height": max(900, min(alto, 3000))})
    pagina.wait_for_timeout(300)
    destino = CAPTURAS / f"caso{caso['id']}.png"
    pagina.screenshot(path=destino)
    pagina.set_viewport_size({"width": 1500, "height": 900})
    print(f"  captura -> {destino.relative_to(RAIZ)}")

    for libro, num_pagina in PAGINAS_PDF.get(caso["id"], []):
        renderizar_pdf(caso["id"], libro, num_pagina)


def main():
    CAPTURAS.mkdir(parents=True, exist_ok=True)
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    solo = {int(a) for a in sys.argv[1:]}  # p. ej. `python pruebas/capturar_pruebas.py 1 5`
    with sync_playwright() as p:
        navegador = p.chromium.launch(channel="chrome")
        pagina = navegador.new_page(viewport={"width": 1500, "height": 900}, locale="es-CO")
        pagina.goto(URL)
        pagina.wait_for_selector("#estado.ok", timeout=60_000)
        for caso in CASOS:
            if not solo or caso["id"] in solo:
                ejecutar_caso(pagina, caso)
        navegador.close()


if __name__ == "__main__":
    main()

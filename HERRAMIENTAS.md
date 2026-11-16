# Herramientas evaluadas por etapa

Este documento recorre cada procedimiento del proyecto (desde la preparación del entorno hasta la interfaz web) y, para cada uno, presenta las alternativas que existían, sus ventajas y desventajas, y la herramienta que finalmente se eligió (marcada con ✅).

Cada sección termina con un espacio de **Justificación** para completar.

## Índice

1. [Gestión del entorno y dependencias](#1-gestión-del-entorno-y-dependencias)
2. [Extracción de texto de los PDF](#2-extracción-de-texto-de-los-pdf)
3. [División en chunks](#3-división-en-chunks)
4. [Almacenamiento intermedio de los chunks](#4-almacenamiento-intermedio-de-los-chunks)
5. [Modelo de embeddings](#5-modelo-de-embeddings)
6. [Librería para generar los embeddings](#6-librería-para-generar-los-embeddings)
7. [Base de datos vectorial](#7-base-de-datos-vectorial)
8. [Métrica de similitud](#8-métrica-de-similitud)
9. [Modelo de lenguaje (LLM) para generar la respuesta](#9-modelo-de-lenguaje-llm-para-generar-la-respuesta)
10. [SDK para llamar al LLM](#10-sdk-para-llamar-al-llm)
11. [Framework de la API](#11-framework-de-la-api)
12. [Servidor de la aplicación](#12-servidor-de-la-aplicación)
13. [Interfaz de usuario (frontend)](#13-interfaz-de-usuario-frontend)

---

## 1. Gestión del entorno y dependencias

**Qué se hizo:** crear un entorno virtual aislado e instalar las librerías del proyecto (`requirements.txt`, con versiones fijadas en `requirements-lock.txt`).

| Herramienta       | Ventajas                                                                                                                                                  | Desventajas                                                                                                      |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| ✅ **uv**         | Muy rápido (escrito en Rust); compatible con la interfaz de `pip` y `venv`; crea entornos e instala en un solo paso; permite generar archivos de bloqueo. | Herramienta relativamente nueva; hay que instalarla aparte; menos documentación y respuestas en foros que `pip`. |
| **pip + venv**    | Viene incluido con Python; es el estándar que todo el mundo conoce; no requiere instalar nada extra.                                                      | Lento al resolver e instalar paquetes pesados (como `torch`); no genera archivo de bloqueo por sí solo.          |
| **conda / mamba** | Gestiona también dependencias que no son de Python (CUDA, librerías de C); popular en ciencia de datos.                                                   | Instalación pesada; entornos grandes; mezclar paquetes de conda y pip puede generar conflictos.                  |
| **Poetry**        | Gestión completa del proyecto (`pyproject.toml`, lock, publicación).                                                                                      | Más configuración de la necesaria para un proyecto de scripts; resolución de dependencias lenta.                 |

**Justificación:**

> _(por completar)_

---

## 2. Extracción de texto de los PDF

**Qué se hizo:** leer cada página de los libros de `pdfs_python/` y obtener su texto junto con el nombre del archivo y el número de página (`procesar_pdfs.py`).

| Herramienta                        | Ventajas                                                                                                                                                                   | Desventajas                                                                                                                   |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| ✅ **PyMuPDF (`fitz`)**            | Muy rápido; extracción por página sencilla; respeta bastante bien saltos de línea e indentación (importante para el código Python); detecta PDF protegidos con contraseña. | No hace OCR por sí solo; licencia AGPL (relevante si el proyecto se distribuyera comercialmente).                             |
| **pypdf**                          | 100 % Python, sin dependencias nativas; licencia permisiva (BSD); fácil de instalar.                                                                                       | Más lento; la calidad de la extracción es peor en PDF con maquetación compleja; a veces pierde espacios o el orden del texto. |
| **pdfplumber / pdfminer.six**      | Gran control sobre la posición de cada carácter; muy bueno para extraer tablas.                                                                                            | Considerablemente más lento en libros de cientos de páginas; API más compleja para un caso simple.                            |
| **Unstructured**                   | Detecta la estructura del documento (títulos, párrafos, listas, tablas); soporta muchos formatos además de PDF.                                                            | Instalación pesada y con muchas dependencias; más lento; más de lo necesario para texto plano.                                |
| **Docling / Marker / PyMuPDF4LLM** | Convierten el PDF a Markdown conservando títulos y bloques de código; resultado muy limpio para RAG.                                                                       | Más lentos (algunos usan modelos de visión y se benefician de GPU); dependencias pesadas; herramientas más jóvenes.           |
| **OCR (Tesseract, EasyOCR)**       | Única opción para PDF escaneados (imágenes sin texto).                                                                                                                     | Muy lento; introduce errores de reconocimiento, sobre todo en código; innecesario si el PDF ya tiene texto.                   |

**Justificación:**

> _(por completar)_

---

## 3. División en chunks

**Qué se hizo:** dividir el texto de cada página con `RecursiveCharacterTextSplitter` de `langchain-text-splitters` (1000 caracteres, 150 de solapamiento), sin mezclar páginas distintas, y fusionar los fragmentos menores de 100 caracteres con un vecino.

| Herramienta / estrategia                                | Ventajas                                                                                                                                                                  | Desventajas                                                                                                                   |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| ✅ **`RecursiveCharacterTextSplitter` (LangChain)**     | Corta primero por párrafos, luego por líneas, luego por palabras, por lo que respeta la estructura del texto; paquete liviano (no requiere todo LangChain); configurable. | Mide en caracteres, no en tokens; no entiende el significado del texto.                                                       |
| **`CharacterTextSplitter`**                             | Muy simple y predecible.                                                                                                                                                  | Usa un único separador, por lo que puede cortar en medio de una idea o de un bloque de código.                                |
| **Splitter por tokens (`TokenTextSplitter`, tiktoken)** | El tamaño coincide exactamente con el límite del modelo de embeddings.                                                                                                    | Depende del tokenizador; más lento; los cortes pueden caer en medio de palabras.                                              |
| **Chunking semántico**                                  | Corta donde cambia el tema, usando embeddings; chunks más coherentes.                                                                                                     | Mucho más lento (requiere calcular embeddings para decidir los cortes); tamaños de chunk irregulares; más difícil de depurar. |
| **`SentenceSplitter` (LlamaIndex)**                     | Respeta los límites de oraciones.                                                                                                                                         | Obliga a usar el ecosistema de LlamaIndex; las "oraciones" no aplican bien al código.                                         |
| **Un chunk por página**                                 | Trivial de implementar; la cita a la página es exacta.                                                                                                                    | Páginas largas exceden lo que el modelo de embeddings procesa bien; páginas cortas generan vectores con poca información.     |
| **Implementación manual completa**                      | Sin dependencias.                                                                                                                                                         | Reimplementar la división recursiva y el solapamiento es propenso a errores.                                                  |

**Justificación:**

> _(por completar)_

---

## 4. Almacenamiento intermedio de los chunks

**Qué se hizo:** guardar todos los chunks con sus metadatos en `chunks.json`, separando la etapa de procesamiento de la de indexación.

| Formato                         | Ventajas                                                                                                    | Desventajas                                                                                                |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| ✅ **JSON**                     | Legible a simple vista; se carga con la librería estándar; conserva la estructura anidada de los metadatos. | Hay que cargar el archivo completo en memoria; ocupa más espacio que un formato binario.                   |
| **JSONL (un objeto por línea)** | Se puede leer y escribir de forma incremental; ideal para archivos muy grandes.                             | Un poco menos cómodo de revisar e inspeccionar manualmente.                                                |
| **CSV**                         | Abre en Excel; muy común.                                                                                   | No maneja bien metadatos anidados ni texto con saltos de línea y comillas (el código Python tiene muchos). |
| **Parquet**                     | Compacto y rápido; buen soporte en pandas.                                                                  | Binario (no se lee a simple vista); requiere `pyarrow`.                                                    |
| **Sin archivo intermedio**      | Un solo script; menos archivos.                                                                             | Cada cambio en la indexación obliga a reprocesar todos los PDF.                                            |

**Justificación:**

> _(por completar)_

---

## 5. Modelo de embeddings

**Qué se hizo:** convertir cada chunk (y cada pregunta) en un vector con `intfloat/multilingual-e5-small`, usando los prefijos `passage:` y `query:` que exige el modelo, normalizando los vectores y usando GPU en float16 cuando está disponible.

| Modelo                                          | Ventajas                                                                                                                                                     | Desventajas                                                                                                                                |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| ✅ **`intfloat/multilingual-e5-small`**         | Multilingüe (preguntas en español sobre libros en inglés o español); pequeño y rápido, funciona en CPU; gratuito y local; buenos resultados en recuperación. | Menor calidad que sus versiones `base` y `large`; límite de 512 tokens por texto; requiere recordar los prefijos `query:`/`passage:`.      |
| **`multilingual-e5-base` / `-large`**           | Mejor calidad de recuperación.                                                                                                                               | Más lentos y pesados; indexar todos los libros tarda bastante más, sobre todo sin GPU.                                                     |
| **`all-MiniLM-L6-v2`**                          | Muy rápido y liviano; muy popular.                                                                                                                           | Entrenado principalmente en inglés; peor con preguntas en español.                                                                         |
| **`paraphrase-multilingual-MiniLM-L12-v2`**     | Multilingüe y liviano.                                                                                                                                       | Pensado para similitud de frases, no específicamente para recuperación de documentos; rinde menos que E5 en búsqueda.                      |
| **`BAAI/bge-m3`**                               | Multilingüe, alta calidad, admite textos largos (hasta 8192 tokens).                                                                                         | Modelo grande (≈ 570 M parámetros); lento sin GPU.                                                                                         |
| **APIs de embeddings (OpenAI, Gemini, Cohere)** | Calidad alta sin necesidad de hardware propio.                                                                                                               | Costo por uso; requieren conexión a internet; envían el contenido a un servicio externo; límites de peticiones al indexar miles de chunks. |

**Justificación:**

> _(por completar)_

---

## 6. Librería para generar los embeddings

**Qué se hizo:** cargar y ejecutar el modelo con `sentence-transformers` (sobre PyTorch).

| Herramienta                                   | Ventajas                                                                                                  | Desventajas                                                             |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| ✅ **sentence-transformers**                  | Una línea para cargar el modelo y otra para codificar; maneja lotes, normalización y GPU automáticamente. | Arrastra PyTorch (descarga pesada, especialmente con CUDA).             |
| **Hugging Face `transformers` directamente**  | Control total del modelo.                                                                                 | Hay que implementar a mano el _pooling_ y la normalización; más código. |
| **FastEmbed / ONNX Runtime**                  | Más liviano que PyTorch; rápido en CPU.                                                                   | Catálogo de modelos más limitado; menos flexible si se quiere usar GPU. |
| **Función de embeddings integrada en Chroma** | Cero código: Chroma calcula los vectores al insertar.                                                     | Menos control sobre el modelo, los prefijos de E5 y el uso de GPU.      |

**Justificación:**

> _(por completar)_

---

## 7. Base de datos vectorial

**Qué se hizo:** guardar los vectores, textos y metadatos en una colección persistente de **Chroma** (`chroma_db/`), con índice HNSW.

| Herramienta                   | Ventajas                                                                                                                                                        | Desventajas                                                                                                 |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| ✅ **Chroma**                 | Se instala con `pip`; funciona embebida en el proceso (sin servidor aparte); persistencia en disco; guarda texto y metadatos junto al vector; API muy sencilla. | No está pensada para volúmenes enormes ni alta concurrencia; menos opciones de ajuste del índice.           |
| **FAISS**                     | Extremadamente rápida; muchos tipos de índices; referencia en investigación.                                                                                    | Es solo un índice de vectores: los textos y metadatos hay que guardarlos aparte; la persistencia es manual. |
| **Qdrant**                    | Filtros por metadatos muy potentes; buen rendimiento; modo local o servidor.                                                                                    | Para usarlo al máximo hay que levantar un servidor (Docker); algo más de configuración.                     |
| **pgvector (PostgreSQL)**     | Une datos relacionales y vectores en una base conocida; SQL estándar.                                                                                           | Requiere instalar y administrar PostgreSQL.                                                                 |
| **Pinecone / Weaviate Cloud** | Servicio administrado y escalable.                                                                                                                              | Requiere cuenta y conexión a internet; costo en planes mayores; los datos quedan en un tercero.             |
| **Milvus**                    | Diseñado para miles de millones de vectores.                                                                                                                    | Despliegue complejo; excesivo para el tamaño de este proyecto.                                              |
| **LanceDB**                   | Embebida como Chroma; formato columnar eficiente.                                                                                                               | Comunidad y documentación más pequeñas.                                                                     |

**Justificación:**

> _(por completar)_

---

## 8. Métrica de similitud

**Qué se hizo:** configurar la colección con distancia **coseno** (`hnsw:space: cosine`) y devolver la similitud como `1 - distancia`.

| Métrica              | Ventajas                                                                                                                                                                    | Desventajas                                                                                                      |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| ✅ **Coseno**        | Compara la dirección de los vectores, no su magnitud; es la métrica recomendada para los modelos E5; el resultado (0 a 1) es fácil de interpretar y mostrar en la interfaz. | Con vectores ya normalizados equivale al producto interno, pero con un pequeño cálculo extra.                    |
| **Euclidiana (L2)**  | Es la métrica por defecto de Chroma.                                                                                                                                        | Sensible a la magnitud de los vectores; la distancia es menos intuitiva para mostrar como porcentaje.            |
| **Producto interno** | La más rápida de calcular.                                                                                                                                                  | Solo es equivalente al coseno si los vectores están normalizados; sin normalizar favorece a los vectores largos. |

**Justificación:**

> _(por completar)_

---

## 9. Modelo de lenguaje (LLM) para generar la respuesta

**Qué se hizo:** usar **Gemini** (por defecto `gemini-3.5-flash`, configurable con `GEMINI_MODEL`) con instrucciones de sistema que lo obligan a responder solo con el contexto y a citar los fragmentos, y con `temperature=0.2`.

| Modelo                                                | Ventajas                                                                                                 | Desventajas                                                                                                                             |
| ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| ✅ **Google Gemini (Flash)**                          | Capa gratuita para desarrollo; rápido; ventana de contexto amplia; buen manejo del español y del código. | Depende de internet y de un servicio externo; límites de peticiones en la capa gratuita; las preguntas y fragmentos se envían a Google. |
| **OpenAI GPT**                                        | Muy buena calidad; ecosistema y documentación enormes.                                                   | De pago desde el primer uso; mismas desventajas de servicio externo.                                                                    |
| **Anthropic Claude**                                  | Muy buena calidad siguiendo instrucciones y citando fuentes.                                             | De pago; mismas desventajas de servicio externo.                                                                                        |
| **Modelos locales (Llama, Mistral, Qwen vía Ollama)** | Gratis; privacidad total; funciona sin internet.                                                         | Requiere hardware potente (idealmente GPU); calidad y velocidad menores en equipos comunes; más configuración.                          |

**Justificación:**

> _(por completar)_

---

## 10. SDK para llamar al LLM

**Qué se hizo:** usar el SDK oficial `google-genai`, pasando el historial del chat y la pregunta con su contexto en el formato de mensajes de Gemini.

| Herramienta                               | Ventajas                                                                                                                   | Desventajas                                                                                     |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| ✅ **`google-genai`**                     | SDK oficial y actual de Google; soporta instrucciones de sistema, historial y configuración de generación de forma tipada. | Atado a Gemini: cambiar de proveedor implica reescribir la llamada.                             |
| **`google-generativeai`**                 | Muchos ejemplos y tutoriales existentes.                                                                                   | SDK anterior, reemplazado por `google-genai`.                                                   |
| **Peticiones HTTP directas (`requests`)** | Sin dependencias extra.                                                                                                    | Hay que construir el JSON y manejar errores a mano.                                             |
| **Wrapper de LangChain / LiteLLM**        | Misma interfaz para varios proveedores.                                                                                    | Capa extra de abstracción y dependencias; puede ir por detrás de las novedades del SDK oficial. |

**Justificación:**

> _(por completar)_

---

## 11. Framework de la API

**Qué se hizo:** exponer la funcionalidad con **FastAPI** (`/api/estado`, `/api/libros`, `/api/buscar`, `/api/preguntar`), validar las peticiones con Pydantic y servir los PDF como archivos estáticos.

| Framework              | Ventajas                                                                                                                       | Desventajas                                                                           |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| ✅ **FastAPI**         | Validación automática de entradas con Pydantic; documentación interactiva en `/docs` (Swagger) sin esfuerzo; moderno y rápido. | Necesita un servidor ASGI aparte (uvicorn); algo más de conceptos que Flask.          |
| **Flask**              | Muy simple y conocido; enorme cantidad de ejemplos.                                                                            | Sin validación ni documentación automática; hay que agregar extensiones.              |
| **Django**             | Incluye ORM, administración, autenticación.                                                                                    | Demasiado grande para una API de pocos endpoints sin base de datos relacional.        |
| **Streamlit / Gradio** | Interfaz y "backend" en un solo script de Python; muy rápido de prototipar.                                                    | No ofrece una API REST reutilizable; control limitado sobre el diseño de la interfaz. |

**Justificación:**

> _(por completar)_

---

## 12. Servidor de la aplicación

**Qué se hizo:** ejecutar la API con `uvicorn api:app --reload`.

| Servidor                          | Ventajas                                                                                 | Desventajas                                                                      |
| --------------------------------- | ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| ✅ **Uvicorn**                    | Es el servidor recomendado por FastAPI; `--reload` reinicia al guardar cambios; liviano. | Para producción con varios procesos se suele combinar con un gestor de procesos. |
| **Gunicorn + workers de Uvicorn** | Varios procesos; robusto en producción.                                                  | No funciona de forma nativa en Windows; innecesario en desarrollo.               |
| **Hypercorn**                     | Soporta HTTP/2 y HTTP/3.                                                                 | Menos usado; menos documentación con FastAPI.                                    |

**Justificación:**

> _(por completar)_

---

## 13. Interfaz de usuario (frontend)

**Qué se hizo:** una página única (`static/index.html`) en HTML, CSS y JavaScript sin framework, servida por la misma API, con un chat y un panel lateral que muestra los fragmentos citados y permite abrir cada PDF en la página correspondiente.

| Herramienta                                    | Ventajas                                                                                                                | Desventajas                                                                                             |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| ✅ **HTML + CSS + JavaScript (sin framework)** | Sin proceso de compilación ni `node_modules`; un solo archivo; control total del diseño; lo sirve FastAPI directamente. | El estado de la interfaz se maneja a mano; el archivo crece a medida que se agregan funciones.          |
| **React / Vue / Svelte**                       | Componentes reutilizables; manejo de estado más ordenado; gran ecosistema.                                              | Requiere Node.js y un paso de _build_; dos proyectos que mantener.                                      |
| **Streamlit**                                  | Interfaz de chat en pocas líneas de Python.                                                                             | Diseño limitado; la página se re-ejecuta en cada interacción; difícil personalizar el panel de fuentes. |
| **Gradio / Chainlit**                          | Componentes de chat ya hechos; pensados para demos de IA.                                                               | Apariencia genérica; menos control sobre cómo mostrar las citas y los PDF.                              |

**Justificación:**

> _(por completar)_

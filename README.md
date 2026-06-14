# 🌌 Multimodal RAG Pipeline

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) 
![Qdrant](https://img.shields.io/badge/Qdrant-DB-FE346E?style=for-the-badge&logo=qdrant&logoColor=white) 
![Gemini](https://img.shields.io/badge/Gemini-AI-8E75B2?style=for-the-badge&logo=google&logoColor=white) 
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)

A highly-optimized, **Multimodal Retrieval-Augmented Generation (RAG)** pipeline designed to ingest, process, and query complex PDF documents. This system goes beyond traditional text-based RAG; it visually understands document layouts, dynamically extracts charts, and retrieves both text and images to provide context-rich, highly accurate answers.

This project showcases an advanced, production-ready approach to Enterprise RAG, moving past naive text chunking to full semantic and visual comprehension.

---

## 🏗️ Architecture Overview

The pipeline is structured into two primary workflows: **Ingestion** and **Retrieval/Generation**.

1. **Ingestion Engine**: 
   - Parses complex PDFs and logically chunks text based on semantic document structure (e.g., font sizes, headers, sub-headers) rather than arbitrary token limits.
   - Visually clusters vector paths to detect, isolate, and extract charts and diagrams.
   - Embeds both text chunks and images into a single, unified multimodal vector space.
2. **Multimodal Retrieval**: 
   - A single natural language query searches the vector space for both relevant text excerpts and relevant charts simultaneously.
   - Retrieved text and high-resolution base64-encoded images are injected directly into a multimodal LLM prompt to synthesize an accurate, well-cited response.

---

## 🧠 Key Design Decisions & Technologies

### 1. Vector Database: Qdrant
We chose **QdrantDB** for its powerful native support for complex multimodal architectures and payload filtering.
- **Simultaneous Multimodal Queries**: Qdrant allows us to run text and image searches *in parallel* with separate scoring thresholds. This ensures that relevant images are never "crowded out" by text chunks in the top-K results.
- **Payload Indexing**: By storing rich metadata (e.g., `modality`, `page_num`, `region_type`, `source`) directly in the vector payload and indexing these fields, Qdrant enables instantaneous filtering (e.g., searching only within "image" modalities) without performance degradation.

### 2. Embedding Model: Gemini Embedding 2.0 (`models/gemini-embedding-2`)
We utilize Google's latest embedding model to create vector representations of our data.
- **Unified Vector Space**: Gemini 2.0 embeddings are natively multimodal. Text and images live in the exact same mathematical space. This is a game-changer: a user can type a text query, and the system seamlessly surfaces relevant images *without* requiring an intermediate, error-prone step to caption or OCR the image first.

### 3. LLM: Gemini 2.5 Flash Lite (`gemini-2.5-flash-lite`)
For synthesis and generation, the pipeline uses Gemini 2.5 Flash Lite orchestrated via LangChain.
- **Native Vision Capabilities**: Instead of relying purely on OCR text extracted from charts, we feed the raw, high-resolution base64 images directly into the LLM alongside the retrieved text excerpts. The model literally "sees" the charts, allowing it to read exact figures, percentages, and visual trends accurately.
- **Speed & Cost-Efficiency**: The "Flash Lite" tier provides the perfect balance of extremely low latency and high reasoning capabilities, making it ideal for snappy, real-time RAG applications.

### 4. Chunking Strategy: Section-Aware & Visual Clustering
Traditional RAG relies on naive token-splitting (e.g., "split every 500 tokens"), which inevitably destroys logical context. We implemented a custom, highly advanced chunking strategy using PyMuPDF (`fitz`):
- **Semantic Text Chunking**: The system dynamically analyzes font sizes across the document to classify text as `page_title`, `section_header`, `sub_header`, or `body`. Text is chunked dynamically based on these logical section boundaries. A chunk only ends when a new section begins, preserving the complete thought and maintaining context.
- **Visual Chart Detection**: Rather than blindly extracting all images, the system uses spatial clustering algorithms on PDF vector drawings. It clusters lines, shapes, and small text elements to intelligently detect entire charts or tables, filtering out decorative lines, headers, and footers. These detected regions are rendered as high-resolution PNGs.
- **Fallback Safety**: For exceptionally long sections (> 2000 tokens), the system gracefully falls back to a recursive character split to stay within model context limits, while still retaining the parent section metadata.

---

## 💼 Why This Matters (Use Cases)
- **Financial Analysis**: Querying annual reports where critical data is locked inside complex charts.
- **Medical Records**: Searching patient histories that include both typed notes and diagnostic imagery.
- **Technical Manuals**: Retrieving instructions that heavily rely on accompanying diagrams and schematics.

---

## 🛠️ Project Structure
- `app/rag/extract.py`: The core ingestion engine handling PDF parsing, font-size classification, and visual clustering.
- `app/rag/embedding.py`: Manages the generation of unified multimodal embeddings via Google GenAI.
- `app/rag/qdrant.py`: Interfaces with Qdrant for creating collections, indexing payloads, and parallel querying.
- `app/rag/retrival.py`: Orchestrates the final retrieval step, combining text and images into a single prompt for the Gemini LLM.

## Run

Windows: .venv\Scripts\activate   
         uvicorn app.main:app --reload 
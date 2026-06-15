# 🌌 RAGfolio: ICICI MF Factsheet RAG Pipeline

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) 
![Qdrant](https://img.shields.io/badge/Qdrant-DB-FE346E?style=for-the-badge&logo=qdrant&logoColor=white) 
![Gemini](https://img.shields.io/badge/Gemini-AI-8E75B2?style=for-the-badge&logo=google&logoColor=white) 
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)

FastAPI backend for **RAGfolio** — a multimodal RAG pipeline that ingests ICICI Prudential mutual fund factsheets and answers queries using both retrieved text chunks and rendered chart images.

🖥️ **Frontend repo:** [RAGfolio-Frontend](https://github.com/jaygajera17/RAGfolio-Frontend)  
🚀 **Live demo:** [ragfolio-frontend.vercel.app](https://ragfolio-frontend.vercel.app)

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

## Key engineering decisions
 
### 1. Section-aware chunking over token splitting
 
Standard RAG splits text every N tokens, which breaks mid-sentence and destroys context. This pipeline uses PyMuPDF to analyse font sizes across the document and classify every text block as `page_title`, `section_header`, `sub_header`, or `body`. Chunks are delimited by section boundaries — a chunk only closes when a new header appears. For sections exceeding 2,000 tokens, a `RecursiveCharacterTextSplitter` fallback is applied *within* the section, preserving the section metadata on every sub-chunk.
 
### 2. Visual chart detection via spatial clustering
 
Instead of extracting all embedded images (which includes logos, banners, decorative lines), the pipeline runs a union-find spatial clustering algorithm on PDF vector drawing paths. Clusters that meet minimum area and element count thresholds are flagged as chart regions and rendered to high-resolution PNGs. This separates meaningful financial charts from page decoration without any ML classifier.
 
### 3. Unified multimodal vector space (Gemini Embedding 2.0)
 
Text chunks and chart images share a single Qdrant collection with a `modality` field. Gemini Embedding 2.0 is natively multimodal — both modalities project into the same 3072-dimension space. A text query can therefore surface relevant charts directly without an intermediate OCR or captioning step.
 
### 4. Separate per-modality retrieval thresholds
 
In a shared embedding space, text-to-text cosine similarity scores (~0.65–0.75) are systematically higher than text-to-image scores (~0.32–0.47). A single unified threshold would either flood results with text and suppress images, or lower the bar enough to return noisy text matches. The solution: two parallel `query_points` calls with separate thresholds (`text: 0.5`, `image: 0.3`), merged before the LLM prompt is built.

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


## Local setup
 
### Prerequisites
 
- Python 3.11+
- A [Qdrant Cloud](https://cloud.qdrant.io/) cluster (free tier works)
- A [Google AI Studio](https://aistudio.google.com/) API key with Gemini access
- An [Auth0](https://auth0.com/) application (Single Page App type)

### 1. Clone and create a virtual environment
 
```bash
git clone https://github.com/jaygajera17/RAGfolio-Backend.git
cd RAGfolio-Backend
python -m venv .venv
 
# Windows
.venv\Scripts\activate
 
# macOS / Linux
source .venv/bin/activate
```
 
### 2. Install dependencies
 
```bash
pip install -r requirements.txt
```
 
### 3. Configure environment variables
 
Create a `.env` file in the project root:
 
```env
# Google Gemini
GOOGLE_API_KEY=your-google-ai-studio-api-key
 
# Qdrant Cloud
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION_NAME=ragfolio
 
# Auth0
AUTH0_DOMAIN=your-tenant.auth0.com
AUTH0_CLIENT_ID=your-auth0-client-id
AUTH0_CLIENT_SECRET=your-auth0-client-secret
APP_BASE_URL=http://127.0.0.1:8000
```
 
| Variable | Where to find it |
|---|---|
| `GOOGLE_API_KEY` | [Google AI Studio](https://aistudio.google.com/) → API keys |
| `QDRANT_URL` | Qdrant Cloud dashboard → your cluster → Endpoint |
| `QDRANT_API_KEY` | Qdrant Cloud dashboard → your cluster → API Keys |
| `AUTH0_DOMAIN` | Auth0 dashboard → Applications → your app → Domain |
| `AUTH0_CLIENT_ID` | Auth0 dashboard → Applications → your app → Client ID |
| `AUTH0_CLIENT_SECRET` | Auth0 dashboard → Applications → your app → Client Secret |
| `APP_BASE_URL` | Must match exactly what is set in Auth0 callback URLs |
 
> **Auth0 callback URLs:** In your Auth0 application settings, set **Allowed Callback URLs** to `http://127.0.0.1:8000/auth/callback` and **Allowed Logout URLs** to `http://127.0.0.1:8000`.

### 4. Run the server
 
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
 
API docs available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
 
### 5. Ingest the factsheet
 
With the server running, trigger ingestion via curl or the `/docs` UI:
 
```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"pdf_path": "static/icici-fund-factsheet-for-may-2026.pdf", "month_year": "May-2026"}'
```

## Deployment
 
The backend is deployed to Vercel as a serverless function. Because Vercel has a 250MB deployment size limit, PDF ingestion dependencies (PyMuPDF, etc.) are stripped from the production bundle — ingestion is intended to be run locally or in a separate worker.
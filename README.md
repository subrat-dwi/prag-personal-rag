# PRAG — Personal RAG System

> A cloud-deployed personal assistant that answers questions about you using your own documents.

## Table of Contents
- [Why PRAG](#why-prag)
- [Demo](#demo-using-prag-web-client)
- [Features](#features)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Setup](#setup)
- [Usage](#usage)
- [Security & Data Privacy](#security--data-privacy)
- [Deployement](#deployment)
- [Contributing](#contributing)

---
## Why PRAG
PRAG solves a specific everyday problem:

```
You're filling a form and can't remember your college roll number.
Someone asks for your internship start date for a reference check.
You want a quick professional intro but don't want to rewrite it from scratch.

Currently you dig through files, photos, WhatsApp, emails.
PRAG lets you just ask.

> "Summarize my technical experience for a job application."
> "What technologies have I worked with across all my projects?"
> "Generate a professional intro I can use in interviews."
> "Which certificate proves my Python knowledge and when did I get it?"
```

---
## Demo (using [PRAG Web Client](https://github.com/subrat-dwi/prag-web))
<p align="center">
  <img src="./demo-media/prag-desktop.png" width="700">
</p>

---

## Features

- **Natural language queries** — ask questions in natural language and get precise answers
- **Hybrid search** — combines semantic vector search with BM25 keyword search to catch both meaning-based and exact matches (scores, IDs, and specific terms)
- **Query intelligence** — spell correction, query rewriting, and classification (factual vs synthesis queries) for improved retrieval and generation
- **Multi-format ingestion** — PDF, DOCX, images (with OCR), Markdown, and plain text
- **OCR support** — extracts text from scanned documents using Tesseract with LLM-based cleaning for noisy output
- **Google Drive sync** — using native `md5Checksum` diffing for incremental re-ingestion and automatic vector cleanup on file deletion
- **Source citations** — every answer cites which document it came from, along with respective drive url
- **Provider-agnostic LLM** — Ollama locally, Groq on deployment — swap with a single env var
- **PWA frontend** — installable on mobile, works like a native app
- **LangSmith tracing** — full observability into prompts, retrieved chunks, and LLM responses (disabled in production)
- **Dockerized Deployment** — Fully containerized with Docker for consistent, one-command setup across local development and production environments.

---

## How It Works

```
Google Drive folder
        │
        ▼
Drive Sync (on startup)
  ├── compare md5Checksum against stored hash in Qdrant
  ├── ingest new / re-ingest changed / remove deleted files
        │
        ▼
Ingestion Pipeline
  parse → chunk → embed → store
  │         │        │       │
  │    RecursiveChar  │    Qdrant
  │    TextSplitter   │    Cloud
  │               fastembed
  │               
  parsers:
  ├── PDF (pypdf + OCR fallback)
  ├── DOCX (python-docx)
  ├── Image (pytesseract)
  ├── Markdown
  └── Plain text
        │
        ▼
User Query (CLI or API)
        │
        ▼
Embed query → Qdrant similarity search → top-k chunks
        │
        ▼
LLM (local / cloud) answers using retrieved context
```

---

## Project Structure

```
prag/
.
├── api
│   ├── __init__.py
│   ├── main.py
│   ├── middleware.py
│   ├── routes
│   │   ├── auth.py
│   │   ├── files.py
│   │   ├── health.py
│   │   ├── query.py
│   │   └── sync.py
│   └── schemas
│       └── query.py
├── cli.py
├── config
│   ├── __init__.py
│   └── settings.py
├── core
│   ├── __init__.py
│   └── rag.py
├── Dockerfile
├── embeddings
│   ├── embedder.py
│   └── __init__.py
├── gdrive_credentials.json
├── ingestion
│   ├── chunker.py
│   ├── drive_sync.py
│   ├── __init__.py
│   ├── parsers
│   │   ├── docx_parser.py
│   │   ├── image_parser.py
│   │   ├── __init__.py
│   │   ├── markdown_parser.py
│   │   ├── pdf_parser.py
│   │   └── text_parser.py
│   └── pipeline.py
├── llm
│   ├── chat_llm.py
│   ├── __init__.py
│   ├── ocr_cleaner_llm.py
│   ├── process_query_llm.py
│   └── utils.py
├── README.md
├── requirements.txt
├── roadmap.md
├── tests
│   └── test_drive.py
└── vectorstore
    ├── __init__.py
    └── qdrant_client.py
```

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM (local) | Any Ollama model depending on your hardware |
| LLM (deployed) | openai/gpt-oss-20b or llama-3.3-70b via Groq API |
| Embeddings | fastembed |
| Vector DB | Qdrant Cloud |
| PDF parsing | pypdf + pymupdf |
| OCR | Tesseract via pytesseract |
| DOCX parsing | python-docx |
| Markdown parsing | markdown + BeautifulSoup |
| LLM Utilities | LangChain |
| Text splitting | LangChain RecursiveCharacterTextSplitter |
| Drive integration | Google Drive API v3 (service account) |
| Config | pydantic-settings |
| Observability | LangSmith |
| API framework | FastAPI |

---

## Setup

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running (if want to run LLM locally)
- Tesseract installed (`sudo apt install tesseract-ocr tesseract-ocr-hin`)
- A [Qdrant Cloud](https://cloud.qdrant.io) free cluster
- A Google Cloud service account with Drive API enabled

### 1. Clone and install

```bash
git clone https://github.com/subrat-dwi/prag-personal-rag
cd prag-personal-rag
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Pull Ollama models

```bash
ollama pull qwen2.5:3b-instruct # or any model
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# LLM
MODEL_PROVIDER=ollama
CHAT_MODEL=qwen2.5:3b-instruct
GROQ_API_KEY=                         # required if MODEL_PROVIDER=groq

# Ollama
OLLAMA_BASE_URL=http://localhost:11434

# Embeddings (set or use default fastembed model)
EMBED_MODEL=

# Qdrant
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=personal-rag
QDRANT_VECTOR_SIZE=384 # or according to your embedding model

# Google Drive
GOOGLE_CREDENTIALS_PATH=gdrive_credentials.json    # only for local working, never push this file
GOOGLE_CREDENTIALS_JSON=              # paste full JSON here for Cloud deployment
DRIVE_FOLDER_ID=your_drive_folder_id

# LangSmith (optional)
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=personal-rag

# Logging
LOG_LEVEL=WARNING
```

### 4. Google Drive setup

1. Create a project in [Google Cloud Console](https://console.cloud.google.com)
2. Enable **Google Drive API**
3. Create a **Service Account** → download `credentials.json` → rename to `gdrive_credentials.json`
4. Share your documents folder in Drive with the service account email (Viewer role)
5. Copy the folder ID from the Drive URL into `DRIVE_FOLDER_ID`

### 5. Run

```bash
python cli.py
```
> **or use with frontend client through API**

Check out [PRAG Web](https://github.com/subrat-dwi/prag-web)

On first run, PRAG syncs your Drive folder, ingests all supported files, and starts the chat interface.

---

## Usage

```
Syncing documents from Google Drive...
Sync complete.

Personal RAG — ask anything about your documents.
Type 'exit' to quit.

You: what is my CGPA
Prag: Your CGPA is 8.36 
[source: Tony_Marksheet.pdf]

You: write an intro for me for backend dev interview
Prag: Hello, My name is...
[source: Tony_Resume.docx]
```

### Adding personal facts directly

Create `.md` or `.txt` files in your Drive folder for information not in your documents:

```markdown
# personal_info.md

Name: Tony Stark
Phone: +91 XXXXXXXXXX
Blood Group: B+
Hobbies: Music, coding, watching movies
```

PRAG will ingest these on next startup and answer questions from them.

### Supported file types

| Format | Parser |
|---|---|
| `.pdf` | pypdf + OCR fallback for scanned pages |
| `.docx` | python-docx |
| `.jpg`, `.jpeg`, `.png` | Tesseract OCR |
| `.md`, `.markdown` | markdown → plain text |
| `.txt`, `.text` | plain text |

---

## Security & Data Privacy

PRAG is a personal productivity tool, not a secure vault.

Your documents stay in Google Drive. Only vector embeddings and 
text chunks are stored in Qdrant Cloud, protected by your API key.

**Suitable for:** resume data, academic records, certificates,
personal facts, contact details, medical history basics.
**Not suitable for:** banking credentials, passwords, or anything 
you wouldn't store in Google Drive.

> If you prefer fully local operation with no cloud dependency,
run Qdrant locally via Docker and use Ollama for LLM —
nothing leaves your machine.

---

## Deployment

PRAG is designed to deploy on [Render](https://render.com) or any cloud hosting platform with Qdrant Cloud for vector storage.

1. Push to GitHub
2. Create a new **Web Service** on Render, connect your repo
3. Set all environment variables from `.env` in Render's dashboard
4. For `GOOGLE_CREDENTIALS_JSON` — paste the entire contents of `gdrive_credentials.json` as a single-line JSON string
5. Switch LLM provider:
```bash
MODEL_PROVIDER=groq
CHAT_MODEL=llama-3.3-70b-versatile # or llama-3.1-8b-instant
GROQ_API_KEY=your_groq_key
```

On every startup, Render will sync your Drive folder and update the index before serving requests.

---
## Contributing

Contributions, suggestions, and feedback are welcome! Whether it's fixing a bug, improving documentation, or proposing a new feature, feel free to open an issue or submit a pull request.

> If you find this project useful or interesting, consider giving it a ⭐ — it helps others discover it and motivates further development.

---
## License

MIT
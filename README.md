# 📚 AI-Based Preparation Guide (RAG)

Your AI-powered study companion for understanding and preparing from your documents.
Upload your materials and get context-aware answers powered by Retrieval-Augmented Generation.

A robust Retrieval-Augmented Generation (RAG) web application that lets users
upload study and reference PDF documents (syllabi, textbook chapters, lecture notes,
research papers, handbooks, etc.) and ask natural-language questions about them. Answers are
generated with Groq LLM (e.g. LLaMA 3.3 70B), supported by exact source citations and optional live internet search enrichment.

If the answer isn't in the documents and no relevant details exist, it responds with:

> "The uploaded documents do not contain this information, and no relevant details were found online."

---

## ✨ Features

- 📄 Multiple PDF upload & indexing
- ✂️ Automatic chunking with overlap for better retrieval
- 🧠 Local HuggingFace Sentence-Transformer embeddings (no API key needed for embeddings)
- 📦 FAISS vector database — saved & loaded locally so you don't have to re-index every run
- 🔎 Configurable top-k similarity retrieval
- 🤖 Groq LLM (e.g., `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`) for ultra-fast, context-grounded answers
- 🌐 Optional Tavily web search integration to supplement answers
- 📚 Source citation (filename + page number) under every answer
- 💬 Chat-style interface with persistent chat history
- ⬇️ Download chat history as a text file
- 🧾 One-click document summary generation
- 🌙 Dark mode toggle
- ⚠️ Robust error handling and loading indicators throughout

---

## 🗂️ Project Structure

```
college-rag-assistant/
├── .vscode/
│   ├── launch.json              # F5 to run Streamlit directly in VS Code
│   ├── settings.json            # Points VS Code at the venv interpreter
│   └── extensions.json          # Recommends the Python extension
├── app.py                       # Streamlit frontend + orchestration
├── utils/
│   ├── __init__.py
│   ├── document_loader.py       # PDF loading & text extraction
│   ├── text_splitter.py         # Chunking documents
│   ├── vector_store.py          # Embeddings + FAISS index management
│   └── rag_chain.py             # Groq-powered RAG answer generation
├── data/
│   ├── uploads/                 # Saved uploaded PDFs (gitignored)
│   └── faiss_index/             # Persisted FAISS index (gitignored)
├── setup.sh                     # One-step venv + dependency installer (Linux/macOS)
├── setup.bat                    # One-step venv + dependency installer (Windows)
├── requirements.txt
├── .env.example                 # Template for your Groq API key
├── .gitignore
└── README.md
```

---

## 🖥️ Quick Start in VS Code

1. Unzip the project and open the folder in VS Code:
   ```bash
   code .
   ```
2. Open a terminal in VS Code (`` Ctrl+` ``) and run the setup script:
   - On Windows:
     ```cmd
     setup.bat
     ```
   - On Linux/macOS:
     ```bash
     chmod +x setup.sh && ./setup.sh
     ```
   This creates `venv/`, installs all dependencies, and creates `.env` for you.
3. Open `.env` and paste in your Groq API key (`GROQ_API_KEY=gsk_...`).
4. Press `Ctrl+Shift+P` → **Python: Select Interpreter** → choose the one inside `venv/`.
5. Run the app either:
   - From the terminal: `streamlit run app.py`, **or**
   - Press **F5** (a pre-configured "Streamlit: Run app.py" launch config is included in `.vscode/launch.json`)
6. Your browser opens automatically at `http://localhost:8501`.

---

## ⚙️ Installation (Manual / Any Editor)

### 1. Clone / unzip the project and move into it
```bash
cd college-rag-assistant
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure your Groq API key
Copy the example env file and add your key:
```bash
cp .env.example .env
```
Then edit `.env`:
```
GROQ_API_KEY=your_actual_groq_api_key_here
```
Get a free key from: https://console.groq.com/keys

> Alternatively, you can paste the API key directly into the sidebar text
> field at runtime if you don't want to use a `.env` file.

### 5. Run the app
```bash
streamlit run app.py
```
Then open the URL Streamlit prints (usually `http://localhost:8501`).

---

## 🚀 How to Use

1. Open the app in your browser.
2. In the sidebar, upload one or more PDF documents.
3. Click **"⚙️ Process & Index Documents"** — this extracts text, chunks
   it, generates embeddings, and builds/updates the local FAISS index.
4. Type your question in the chat box at the bottom (e.g. *"What is the
   eligibility criteria for the B.Tech CSE program?"*).
5. The assistant retrieves the most relevant chunks and asks Groq to
   answer using only that context, showing **Sources** under the answer.
6. Use the sidebar to adjust **top-k**, select your **Groq model**, toggle **dark mode**, generate a
   **document summary**, or **download your chat history**.

---

## 🧩 How It Works (RAG Pipeline)

1. **Document Loading** (`document_loader.py`) — `PyPDFLoader` extracts
   text per page from each uploaded PDF, tagging each page with its source
   filename and page number.
2. **Chunking** (`text_splitter.py`) — `RecursiveCharacterTextSplitter`
   breaks pages into ~1000-character chunks with 150-character overlap so
   retrieval stays precise while preserving context across boundaries.
3. **Embedding & Indexing** (`vector_store.py`) — Each chunk is embedded
   using the `sentence-transformers/all-MiniLM-L6-v2` model and stored in
   a FAISS vector index, which is saved to `data/faiss_index/` so it
   persists across app restarts.
4. **Retrieval** — On each question, FAISS similarity search returns the
   top-k most relevant chunks.
5. **Generation** (`rag_chain.py`) — The retrieved chunks are inserted
   into a strict prompt sent to Groq, instructing it to answer **only**
   from that context, and to use the exact fallback sentence when the
   answer isn't present.

---

## 🛠️ Tech Stack

| Layer            | Technology                                   |
|-------------------|-----------------------------------------------|
| Frontend          | Streamlit                                     |
| Orchestration     | LangChain                                     |
| PDF Parsing       | pypdf / PyPDFLoader                           |
| Embeddings        | HuggingFace Sentence Transformers (MiniLM-L6) |
| Vector Database   | FAISS (local, persisted to disk)              |
| LLM               | Groq (`llama-3.3-70b-versatile`)              |

---

## 🖼️ Scanned PDFs (OCR Setup)

If your PDF is a **scanned document** (a picture of text rather than real
selectable text), `pypdf` alone can't extract anything from it. This app
automatically detects pages with little/no extractable text and falls back
to OCR (Optical Character Recognition) — but two external programs must be
installed first.

**How to tell if your PDF is scanned:** open it in any PDF viewer and try
to select/copy a line of text with your mouse. If nothing highlights, it's
scanned.

### 1. Install Tesseract OCR
- **Windows:** download and run the installer from
  https://github.com/UB-Mannheim/tesseract/wiki
- **Mac:** `brew install tesseract`
- **Linux:** `sudo apt install tesseract-ocr`

### 2. Install Poppler (needed to convert PDF pages into images)
- **Windows:** download from
  https://github.com/oschwartz10612/poppler-windows/releases, unzip it
  anywhere (e.g. `C:\poppler-24.08.0`)
- **Mac:** `brew install poppler`
- **Linux:** `sudo apt install poppler-utils`

### 3. Tell the app where to find them (Windows only, if not on PATH)
Open `.env` and uncomment/fill these two lines with your actual install
paths:
```
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
POPPLER_PATH=C:\poppler-24.08.0\Library\bin
```
(Mac/Linux installs via `brew`/`apt` are usually already on PATH and don't
need this step.)

### 4. Re-process your PDF
Re-upload and click **"⚙️ Process & Index Documents"** again. Scanned pages
will now be OCR'd automatically — no other steps needed. If OCR succeeds,
indexing will report real chunk counts instead of effectively-empty text.

> Note: OCR is slower than normal text extraction and accuracy depends on
> scan quality. For best results, use PDFs with clean, high-contrast,
> upright scanned pages.

---

## 🐛 Troubleshooting

- **"Please provide your Groq API key"** — Set `GROQ_API_KEY` in
  `.env` or paste it into the sidebar field.
- **`allow_dangerous_deserialization` errors** — This app only loads
  FAISS indexes it created itself; this flag is expected and safe in this
  context.
- **Slow first run** — The embedding model downloads (~80MB) the first
  time it's used; subsequent runs are fast.
- **Out-of-date answers after re-uploading a file with the same name** —
  Click "🗑️ Clear Chat History" and re-process; for a full reset, delete
  the `data/faiss_index/` folder and re-index.

---

## 📌 Notes on Production Readiness

- Add authentication if deploying publicly (Streamlit alone has no auth).
- Consider chunked, async embedding for very large PDF batches.
- Rotate/secure your Groq API key; never commit `.env` to version control.

"""
app.py
------
Main Streamlit entry point for the AI-Based Preparation Guide
(RAG application) powered by Groq.
"""
# Eagerly initialize concurrent.futures modules to prevent Python 3.13 atexit shutdown bug on Windows
import concurrent.futures.process
import concurrent.futures.thread

import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from utils.document_loader import (
    count_pages_needing_ocr,
    load_multiple_pdfs,
    save_uploaded_file,
)
from utils.rag_chain import answer_question, get_llm, summarize_documents
from utils.text_splitter import split_documents
from utils.vector_store import (
    add_documents_to_store,
    build_vector_store,
    clear_vector_store,
    get_embedding_model,
    get_indexed_filenames,
    load_vector_store,
    save_vector_store,
    similarity_search,
)

# --------------------------------------------------------------------------
# Configuration & setup
# --------------------------------------------------------------------------
# FORCE path check: Find .env explicitly where app.py is located
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
env_path = current_dir / ".env"
load_dotenv(dotenv_path=env_path)

UPLOAD_DIR = "data/uploads"
INDEX_DIR = "data/faiss_index"

st.set_page_config(
    page_title="AI-Based Preparation Guide",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Cached resources (loaded once per session / process)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def cached_embedding_model():
    """Cache the HuggingFace embedding model so it's loaded only once."""
    return get_embedding_model()


@st.cache_data(ttl=300, show_spinner=False)
def get_available_groq_models(api_key: str):
    """Dynamically query Groq for available chat models on this account."""
    fallback_models = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
        "allam-2-7b",
    ]
    if not api_key:
        return fallback_models
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        models_data = client.models.list().data
        chat_models = [
            m.id for m in models_data
            if not m.id.startswith("whisper") and "guard" not in m.id and "orpheus" not in m.id
        ]
        return chat_models if chat_models else fallback_models
    except Exception:
        return fallback_models


def get_api_key() -> str:
    """Resolve the Groq API key from env var or sidebar input."""
    env_key = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
    return env_key or st.session_state.get("api_key_input", "")


# --------------------------------------------------------------------------
# Session state initialization
# --------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {role, content, sources, time}

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


# --------------------------------------------------------------------------
# Theme styling (Light & Dark Mode with optimal contrast & visual polish)
# --------------------------------------------------------------------------
def apply_theme(dark: bool) -> None:
    """Inject custom CSS ensuring high-contrast readability and modern aesthetics in both Light and Dark modes."""
    if dark:
        theme_vars = """
        --bg-page: #0b0f19;
        --bg-surface: #131b2e;
        --bg-surface-elevated: #1a243d;
        --bg-sidebar: #0e1424;
        --border-subtle: #1e293b;
        --border-medium: #334155;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --accent-primary: #3b82f6;
        --accent-hover: #60a5fa;
        --accent-light: rgba(59, 130, 246, 0.15);
        --accent-gradient: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
        --chat-user-bg: #1e293b;
        --chat-user-border: #3b82f6;
        --chat-user-text: #f8fafc;
        --chat-ai-bg: #131b2e;
        --chat-ai-border: #1e293b;
        --chat-ai-text: #f8fafc;
        --card-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
        --card-hover-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
        --badge-ready-bg: rgba(16, 185, 129, 0.15);
        --badge-ready-text: #34d399;
        --badge-ready-border: rgba(16, 185, 129, 0.3);
        --code-bg: #0a0e17;
        --code-text: #38bdf8;

        /* Upload document */
        --uploader-bg: #131b2e;
        --uploader-border: #334155;
        --uploader-text: #f8fafc;
        --uploader-btn-bg: #1e293b;
        --uploader-btn-text: #f8fafc;
        --uploader-btn-border: #475569;

        /* Around chat area & input */
        --bottom-bg: #0b0f19;
        --bottom-border: #1e293b;
        --chat-input-bg: #131b2e;
        --chat-input-border: #334155;
        --chat-input-text: #f8fafc;
        --chat-input-placeholder: #94a3b8;
        --chat-input-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);

        /* Buttons (including Download Chat History) */
        --btn-secondary-bg: #1e293b;
        --btn-secondary-text: #f8fafc;
        --btn-secondary-border: #334155;
        --btn-secondary-hover-bg: #223049;
        --btn-secondary-hover-text: #60a5fa;

        /* Sources */
        --source-box-bg: #0d1527;
        --source-box-border: #1e293b;
        --source-summary-bg: #0d1527;
        --source-summary-text: #f8fafc;
        --source-details-bg: #131b2e;
        --citation-pill-bg: #1e293b;
        --citation-pill-border: #334155;
        --citation-pill-text: #f8fafc;
        --citation-badge-bg: #1e3a5f;
        --citation-badge-text: #93c5fd;
        --citation-badge-border: #2563eb;
        """
    else:
        theme_vars = """
        --bg-page: #f8fafc;
        --bg-surface: #ffffff;
        --bg-surface-elevated: #f1f5f9;
        --bg-sidebar: #ffffff;
        --border-subtle: #e2e8f0;
        --border-medium: #cbd5e1;
        --text-primary: #0f172a;
        --text-secondary: #334155;
        --text-muted: #64748b;
        --accent-primary: #2563eb;
        --accent-hover: #1d4ed8;
        --accent-light: #eff6ff;
        --accent-gradient: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%);
        --chat-user-bg: #eff6ff;
        --chat-user-border: #bfdbfe;
        --chat-user-text: #1e3a8a;
        --chat-ai-bg: #ffffff;
        --chat-ai-border: #e2e8f0;
        --chat-ai-text: #0f172a;
        --card-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        --card-hover-shadow: 0 8px 20px rgba(0, 0, 0, 0.09);
        --badge-ready-bg: #ecfdf5;
        --badge-ready-text: #065f46;
        --badge-ready-border: #a7f3d0;
        --code-bg: #f8fafc;
        --code-text: #0f172a;

        /* Upload document - Clean bright light mode */
        --uploader-bg: #f8fafc;
        --uploader-border: #cbd5e1;
        --uploader-text: #0f172a;
        --uploader-btn-bg: #ffffff;
        --uploader-btn-text: #0f172a;
        --uploader-btn-border: #cbd5e1;

        /* Around chat area & input - Bright light mode */
        --bottom-bg: #f8fafc;
        --bottom-border: #e2e8f0;
        --chat-input-bg: #ffffff;
        --chat-input-border: #cbd5e1;
        --chat-input-text: #0f172a;
        --chat-input-placeholder: #64748b;
        --chat-input-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);

        /* Buttons (including Download Chat History) - Clean white with dark readable text */
        --btn-secondary-bg: #ffffff;
        --btn-secondary-text: #0f172a;
        --btn-secondary-border: #cbd5e1;
        --btn-secondary-hover-bg: #eff6ff;
        --btn-secondary-hover-text: #2563eb;

        /* Sources - Clean light card with dark readable text */
        --source-box-bg: #f8fafc;
        --source-box-border: #e2e8f0;
        --source-summary-bg: #f8fafc;
        --source-summary-text: #0f172a;
        --source-details-bg: #ffffff;
        --citation-pill-bg: #f1f5f9;
        --citation-pill-border: #cbd5e1;
        --citation-pill-text: #0f172a;
        --citation-badge-bg: #dbeafe;
        --citation-badge-text: #1e40af;
        --citation-badge-border: #bfdbfe;
        """

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {{
        {theme_vars}
    }}

    /* Global Typography & App Styling */
    html, body, [class*="css"], .stApp {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        background-color: var(--bg-page) !important;
        color: var(--text-primary) !important;
    }}

    /* Constrain main block width to eliminate large unused empty space */
    .main .block-container {{
        max-width: 980px !important;
        padding-top: 1.5rem !important;
        padding-bottom: 5.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }}
    section[data-testid="stSidebar"] .block-container {{
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }}
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown {{
        color: var(--text-primary) !important;
    }}
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {{
        color: var(--text-primary) !important;
    }}
    section[data-testid="stSidebar"] .stCaption p {{
        color: var(--text-muted) !important;
    }}

    /* Headings */
    h1, h2, h3, h4, h5, h6 {{
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }}

    /* Text & Paragraphs */
    .stMarkdown p, .stMarkdown li {{
        color: var(--text-primary) !important;
        line-height: 1.68 !important;
    }}

    /* Chat Messages - Light & Dark Mode High Contrast */
    [data-testid="stChatMessage"] {{
        background-color: var(--chat-ai-bg) !important;
        border: 1px solid var(--chat-ai-border) !important;
        border-radius: 14px !important;
        padding: 1.25rem 1.4rem !important;
        margin-bottom: 1.25rem !important;
        box-shadow: var(--card-shadow) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }}
    [data-testid="stChatMessage"]:hover {{
        box-shadow: var(--card-hover-shadow) !important;
    }}
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] div,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] td,
    [data-testid="stChatMessage"] th {{
        color: var(--text-primary) !important;
    }}
    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] b {{
        color: var(--text-primary) !important;
        font-weight: 700 !important;
    }}

    /* User Message distinct background */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]),
    [data-testid="stChatMessage"][data-testid*="user"] {{
        background-color: var(--chat-user-bg) !important;
        border: 1px solid var(--chat-user-border) !important;
    }}
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p,
    [data-testid="stChatMessage"][data-testid*="user"] p {{
        color: var(--chat-user-text) !important;
        font-weight: 500 !important;
    }}

    /* Code Blocks */
    code, pre {{
        font-family: 'JetBrains Mono', Consolas, Monaco, monospace !important;
    }}
    code:not(pre code) {{
        background-color: var(--accent-light) !important;
        color: var(--accent-primary) !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 6px !important;
        font-size: 0.88em !important;
        border: 1px solid var(--border-subtle) !important;
    }}
    pre {{
        background-color: var(--code-bg) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 10px !important;
        padding: 1rem !important;
    }}
    pre code {{
        color: var(--code-text) !important;
    }}

    /* ========================================================================== */
    /* 1. UPLOAD DOCUMENT STYLING (FIX FOR LIGHT MODE BLACK COLOR)                */
    /* ========================================================================== */
    [data-testid="stFileUploader"] {{
        background-color: transparent !important;
    }}
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploaderDropzone"] {{
        background-color: var(--uploader-bg) !important;
        border: 2px dashed var(--uploader-border) !important;
        border-radius: 14px !important;
        padding: 1.25rem 1rem !important;
        transition: all 0.2s ease !important;
    }}
    [data-testid="stFileUploader"] section:hover,
    [data-testid="stFileUploaderDropzone"]:hover {{
        border-color: var(--accent-primary) !important;
        background-color: var(--accent-light) !important;
    }}
    [data-testid="stFileUploader"] section small,
    [data-testid="stFileUploader"] section span,
    [data-testid="stFileUploader"] section p,
    [data-testid="stFileUploader"] section div,
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small,
    [data-testid="stFileUploaderDropzoneInstructions"] div {{
        color: var(--uploader-text) !important;
    }}
    /* Browse Files button inside uploader */
    [data-testid="stFileUploader"] button,
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploader"] [data-testid="baseButton-secondary"] {{
        background-color: var(--uploader-btn-bg) !important;
        color: var(--uploader-btn-text) !important;
        border: 1.5px solid var(--uploader-btn-border) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
        transition: all 0.2s ease !important;
    }}
    [data-testid="stFileUploader"] button:hover,
    [data-testid="stFileUploaderDropzone"] button:hover {{
        background-color: var(--accent-light) !important;
        border-color: var(--accent-primary) !important;
        color: var(--accent-primary) !important;
    }}
    /* Uploaded file list items */
    [data-testid="stFileUploaderFile"] {{
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
        padding: 0.5rem 0.75rem !important;
    }}
    [data-testid="stFileUploaderFile"] span,
    [data-testid="stFileUploaderFile"] small,
    [data-testid="stFileUploaderFile"] div {{
        color: var(--text-primary) !important;
    }}

    /* ========================================================================== */
    /* 2. AROUND CHAT AREA STYLING (FIX FOR LIGHT MODE BLACK COLOR)               */
    /* ========================================================================== */
    [data-testid="stBottom"],
    [data-testid="stBottomBlockContainer"],
    .stChatFloatingInputContainer,
    footer {{
        background-color: var(--bottom-bg) !important;
        background: var(--bottom-bg) !important;
    }}
    [data-testid="stBottom"] {{
        border-top: 1px solid var(--bottom-border) !important;
    }}
    [data-testid="stChatInput"] {{
        background-color: transparent !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }}
    [data-testid="stChatInput"] > div {{
        border-radius: 24px !important;
        border: 1.5px solid var(--chat-input-border) !important;
        background-color: var(--chat-input-bg) !important;
        box-shadow: var(--chat-input-shadow) !important;
        transition: all 0.2s ease !important;
        padding: 0.25rem 0.5rem !important;
    }}
    [data-testid="stChatInput"] > div:focus-within {{
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 3px var(--accent-light), var(--chat-input-shadow) !important;
    }}
    [data-testid="stChatInput"] textarea {{
        background-color: transparent !important;
        color: var(--chat-input-text) !important;
        font-size: 0.96rem !important;
        font-weight: 400 !important;
    }}
    [data-testid="stChatInput"] textarea::placeholder {{
        color: var(--chat-input-placeholder) !important;
    }}
    [data-testid="stChatInput"] button {{
        background-color: var(--accent-primary) !important;
        color: #ffffff !important;
        border-radius: 50% !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }}
    [data-testid="stChatInput"] button:hover {{
        background-color: var(--accent-hover) !important;
        transform: scale(1.06) !important;
    }}
    [data-testid="stChatInput"] button svg {{
        fill: #ffffff !important;
        stroke: #ffffff !important;
    }}

    /* ========================================================================== */
    /* 3. BUTTONS & DOWNLOAD CHAT HISTORY (FIX FOR LIGHT MODE BLACK COLOR)        */
    /* ========================================================================== */
    .stButton > button,
    .stDownloadButton > button,
    [data-testid="stDownloadButton"] > button,
    [data-testid="stBaseButton-secondary"] {{
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.15rem !important;
        border: 1.5px solid var(--btn-secondary-border) !important;
        background-color: var(--btn-secondary-bg) !important;
        color: var(--btn-secondary-text) !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }}
    .stButton > button:hover,
    .stDownloadButton > button:hover,
    [data-testid="stDownloadButton"] > button:hover,
    [data-testid="stBaseButton-secondary"]:hover {{
        border-color: var(--accent-primary) !important;
        color: var(--btn-secondary-hover-text) !important;
        background-color: var(--btn-secondary-hover-bg) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08) !important;
    }}
    /* Primary buttons */
    .stButton > button[kind="primary"],
    [data-testid="stBaseButton-primary"] {{
        background: var(--accent-gradient) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25) !important;
    }}
    .stButton > button[kind="primary"]:hover,
    [data-testid="stBaseButton-primary"]:hover {{
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.4) !important;
        transform: translateY(-1.5px) !important;
        color: #ffffff !important;
    }}

    /* ========================================================================== */
    /* 4. SOURCES EXPANDER & CITATIONS (FIX FOR LIGHT MODE BLACK COLOR)           */
    /* ========================================================================== */
    [data-testid="stExpander"],
    details[data-testid="stExpander"] {{
        background-color: var(--source-box-bg) !important;
        border: 1.5px solid var(--source-box-border) !important;
        border-radius: 12px !important;
        margin-top: 0.85rem !important;
        box-shadow: none !important;
        overflow: hidden !important;
    }}
    [data-testid="stExpander"] summary,
    details[data-testid="stExpander"] summary {{
        background-color: var(--source-summary-bg) !important;
        color: var(--source-summary-text) !important;
        font-weight: 700 !important;
        padding: 0.65rem 0.85rem !important;
        border-radius: 10px !important;
        transition: background-color 0.2s ease !important;
    }}
    [data-testid="stExpander"] summary:hover,
    details[data-testid="stExpander"] summary:hover {{
        background-color: var(--accent-light) !important;
        color: var(--accent-primary) !important;
    }}
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span {{
        color: var(--source-summary-text) !important;
        font-weight: 700 !important;
    }}
    [data-testid="stExpander"] summary svg,
    details[data-testid="stExpander"] summary svg {{
        fill: var(--source-summary-text) !important;
        stroke: var(--source-summary-text) !important;
        color: var(--source-summary-text) !important;
    }}
    [data-testid="stExpander"] [data-testid="stExpanderDetails"],
    details[data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
        background-color: var(--source-details-bg) !important;
        border-top: 1px solid var(--source-box-border) !important;
        border-radius: 0 0 12px 12px !important;
        padding: 0.85rem 1rem !important;
    }}
    .source-citation-item {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: var(--citation-pill-bg) !important;
        border: 1.5px solid var(--citation-pill-border) !important;
        border-radius: 8px !important;
        padding: 0.45rem 0.75rem !important;
        margin: 0.25rem 0.35rem 0.25rem 0 !important;
        font-size: 0.84rem !important;
        font-weight: 600 !important;
        color: var(--citation-pill-text) !important;
        transition: all 0.2s ease !important;
    }}
    .source-citation-item:hover {{
        border-color: var(--accent-primary) !important;
        color: var(--accent-primary) !important;
        background: var(--accent-light) !important;
    }}
    .source-page-badge {{
        font-size: 0.74rem !important;
        background: var(--citation-badge-bg) !important;
        color: var(--citation-badge-text) !important;
        padding: 0.15rem 0.5rem !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
        border: 1px solid var(--citation-badge-border) !important;
    }}

    /* Selectbox, Text input, Radio, Slider */
    .stTextInput input, [data-baseweb="select"] {{
        border-radius: 10px !important;
        background-color: var(--bg-surface) !important;
        color: var(--text-primary) !important;
        border: 1.5px solid var(--border-medium) !important;
    }}
    [data-baseweb="select"] * {{
        color: var(--text-primary) !important;
    }}

    /* Custom Header Container */
    .guide-header-card {{
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 1.75rem 2rem;
        margin-bottom: 1.75rem;
        box-shadow: var(--card-shadow);
        position: relative;
        overflow: hidden;
    }}
    .guide-header-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--accent-gradient);
    }}
    .guide-header-title {{
        font-size: 1.85rem;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: -0.025em;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }}
    .guide-header-desc {{
        font-size: 0.98rem;
        color: var(--text-secondary);
        line-height: 1.55;
        margin-bottom: 1rem;
    }}
    .guide-meta-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        align-items: center;
        padding-top: 0.6rem;
        border-top: 1px solid var(--border-subtle);
    }}
    .guide-meta-chip {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        background-color: var(--bg-surface-elevated);
        color: var(--text-primary);
        border: 1px solid var(--border-subtle);
    }}
    .guide-meta-chip.ready {{
        background-color: var(--badge-ready-bg);
        color: var(--badge-ready-text);
        border-color: var(--badge-ready-border);
    }}

    /* Welcome / Empty State Grid */
    .welcome-container {{
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 20px;
        padding: 2.25rem 2rem;
        margin-bottom: 1.75rem;
        box-shadow: var(--card-shadow);
        text-align: center;
    }}
    .welcome-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        background: var(--accent-light);
        color: var(--accent-primary);
        margin-bottom: 1rem;
    }}
    .welcome-title {{
        font-size: 2rem;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: -0.03em;
        margin-bottom: 0.5rem;
    }}
    .welcome-subtitle {{
        font-size: 1.05rem;
        color: var(--text-secondary);
        margin-bottom: 1.75rem;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }}
    .feature-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.1rem;
        text-align: left;
        margin-bottom: 0.5rem;
    }}
    .feature-card {{
        background: var(--bg-surface-elevated);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 1.25rem 1.15rem;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    .feature-card:hover {{
        transform: translateY(-3px);
        box-shadow: var(--card-hover-shadow);
        border-color: var(--accent-primary);
    }}
    .feature-icon {{
        font-size: 1.6rem;
        margin-bottom: 0.6rem;
        display: inline-block;
    }}
    .feature-title {{
        font-size: 0.98rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.35rem;
    }}
    .feature-desc {{
        font-size: 0.83rem;
        color: var(--text-secondary);
        line-height: 1.45;
    }}

    /* Document card in sidebar */
    .doc-badge-item {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        transition: all 0.2s ease;
    }}
    .doc-badge-item:hover {{
        border-color: var(--accent-primary);
    }}
    .doc-badge-name {{
        font-size: 0.84rem;
        font-weight: 600;
        color: var(--text-primary);
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        max-width: 170px;
    }}
    .doc-badge-pill {{
        font-size: 0.7rem;
        font-weight: 700;
        padding: 0.2rem 0.5rem;
        border-radius: 9999px;
        background: var(--badge-ready-bg);
        color: var(--badge-ready-text);
        border: 1px solid var(--badge-ready-border);
    }}

    /* Message timestamp */
    .msg-timestamp {{
        font-size: 0.73rem;
        color: var(--text-muted);
        margin-top: 0.5rem;
        text-align: right;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# Apply the theme immediately
apply_theme(st.session_state.dark_mode)


# --------------------------------------------------------------------------
# Sidebar: configuration, upload, indexing, bonus features
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("📚 AI-Based Preparation Guide")
    st.caption("Document Processing & Settings")

    # --- API key box visibility handling ---
    resolved_key = get_api_key()
    if not resolved_key:
        st.text_input(
            "Groq API Key",
            type="password",
            key="api_key_input",
            help="Get your key at https://console.groq.com/keys or set GROQ_API_KEY in .env",
        )
    else:
        st.success("🔒 Groq API Key loaded from system configuration.")

    # --- Groq model selection ---
    available_models = get_available_groq_models(resolved_key)
    env_default = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-120b").strip()
    default_idx = available_models.index(env_default) if env_default in available_models else 0

    selected_model = st.selectbox(
        "🤖 Groq Model",
        options=available_models,
        index=default_idx,
        help="Select the Groq model to use for answering questions and summarization.",
    )

    st.divider()

    # --- Dark mode toggle ---
    dark_mode_val = st.toggle("🌙 Dark mode", value=st.session_state.dark_mode)
    if dark_mode_val != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode_val
        st.rerun()

    st.divider()

    # --- PDF upload (multiple PDFs supported) ---
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    index_mode = st.radio(
        "📁 Indexing Strategy",
        options=[
            "🔄 Start Fresh (replace previous documents)",
            "➕ Append (add to previous documents)",
        ],
        index=0,
        help="Select 'Start Fresh' to clear previous documents and index only the files currently uploaded.",
    )

    # --- Top-k retrieval configuration ---
    top_k = st.slider(
        "🔎 Number of chunks to retrieve (top-k)",
        min_value=1,
        max_value=10,
        value=4,
        help="Higher values give the LLM more context but may add noise.",
    )

    process_clicked = st.button("⚙️ Process & Index Documents", use_container_width=True, type="primary")

    if st.button("🗑️ Reset & Clear Indexed Documents", use_container_width=True):
        clear_vector_store(INDEX_DIR)
        st.session_state.vector_store = None
        st.session_state.indexed_files = set()
        st.session_state.chat_history = []
        st.success("✅ Knowledge base cleared successfully! Old documents removed.")
        st.rerun()

    # Try to load an existing index from disk on first run, if present.
    if st.session_state.vector_store is None:
        existing = load_vector_store(INDEX_DIR, cached_embedding_model())
        if existing is not None:
            st.session_state.vector_store = existing

    if process_clicked:
        if not uploaded_files:
            st.warning("Please upload at least one PDF file first.")
        else:
            with st.spinner("📚 Reading PDFs, splitting into chunks, and building embeddings..."):
                try:
                    # If Start Fresh is chosen, clear previous index first!
                    if index_mode.startswith("🔄 Start Fresh"):
                        clear_vector_store(INDEX_DIR)
                        st.session_state.vector_store = None
                        st.session_state.indexed_files = set()

                    # 1. Save uploads to disk
                    saved_paths = [
                        save_uploaded_file(f, UPLOAD_DIR) for f in uploaded_files
                    ]

                    # 2. Extract text from PDFs
                    documents = load_multiple_pdfs(saved_paths)

                    # 3. Split into chunks
                    chunks = split_documents(documents)

                    # 4. Embed & store in FAISS
                    embeddings = cached_embedding_model()
                    if st.session_state.vector_store is None:
                        st.session_state.vector_store = build_vector_store(
                            chunks, embeddings
                        )
                    else:
                        add_documents_to_store(
                            st.session_state.vector_store, chunks
                        )

                    # 5. Persist index locally
                    save_vector_store(st.session_state.vector_store, INDEX_DIR)

                    indexed_list = get_indexed_filenames(st.session_state.vector_store)
                    st.session_state.indexed_files = set(indexed_list)
                    st.success(
                        f"✅ Indexed {len(uploaded_files)} file(s) into "
                        f"{len(chunks)} chunks successfully!"
                    )

                    ocr_pages = count_pages_needing_ocr(documents)
                    if ocr_pages:
                        st.warning(
                            f"⚠️ {ocr_pages} page(s) appear to be scanned "
                            "images with little/no extractable text, and "
                            "OCR is not set up on this machine, so those "
                            "pages were indexed with empty content. See "
                            "the README's 'Scanned PDFs' section to "
                            "enable OCR support."
                        )
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error while processing documents: {e}")

    # Display currently active documents dynamically
    active_docs = get_indexed_filenames(st.session_state.vector_store)
    if active_docs:
        st.divider()
        st.subheader("📑 Active Documents")
        for fname in active_docs:
            st.markdown(
                f"""
                <div class="doc-badge-item">
                    <span class="doc-badge-name" title="{fname}">📄 {fname}</span>
                    <span class="doc-badge-pill">Indexed</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.divider()
        st.caption("ℹ️ No documents currently indexed.")

    # --- Document summary generation (bonus) ---
    st.divider()
    if st.button("🧾 Generate Document Summary", use_container_width=True):
        api_key = get_api_key()
        if not api_key:
            st.warning("Please provide your Groq API key first.")
        elif st.session_state.vector_store is None:
            st.warning("Please upload and process documents first.")
        else:
            with st.spinner("Generating summary..."):
                try:
                    llm = get_llm(api_key, model_name=selected_model)
                    # Pull a broad sample of chunks via a generic query to summarize.
                    sample_docs = similarity_search(
                        st.session_state.vector_store, "overview of the document", k=20
                    )
                    summary = summarize_documents(llm, sample_docs)
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": f"**📋 Document Summary:**\n\n{summary}",
                            "sources": [],
                            "time": datetime.now().strftime("%H:%M:%S"),
                        }
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error generating summary: {e}")

    # --- Download chat history (bonus) ---
    st.divider()
    if st.session_state.chat_history:
        history_text = "\n\n".join(
            f"[{m['time']}] {m['role'].upper()}: {m['content']}"
            for m in st.session_state.chat_history
        )
        st.download_button(
            "⬇️ Download Chat History",
            data=history_text,
            file_name="chat_history.txt",
            mime="text/plain",
            use_container_width=True,
        )

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# --------------------------------------------------------------------------
# Main Chat Interface
# --------------------------------------------------------------------------

# Header Card
current_active_docs = get_indexed_filenames(st.session_state.vector_store)
doc_count = len(current_active_docs)
has_tavily = bool(os.getenv("TAVILY_API_KEY"))

st.markdown(
    f"""
    <div class="guide-header-card">
        <div class="guide-header-title">📚 AI-Based Preparation Guide</div>
        <div class="guide-header-desc">
            Your AI-powered study companion for understanding and preparing from your documents.<br>
            Upload your materials and get context-aware answers powered by Retrieval-Augmented Generation.
        </div>
        <div class="guide-meta-row">
            <span class="guide-meta-chip {'ready' if doc_count > 0 else ''}">
                {'🟢 ' + str(doc_count) + ' Document(s) Indexed' if doc_count > 0 else '⚪ No Documents Uploaded'}
            </span>
            <span class="guide-meta-chip">🤖 Model: {selected_model}</span>
            <span class="guide-meta-chip">🔎 Retrieval: Top-{top_k} Chunks</span>
            <span class="guide-meta-chip">{'🌐 Web Search Active' if has_tavily else '🔒 Local Document RAG'}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Empty state hero when no messages exist
if not st.session_state.chat_history:
    st.markdown(
        """
        <div class="welcome-container">
            <div class="welcome-badge">✨ Interactive RAG Assistant</div>
            <div class="welcome-title">👋 Welcome to AI-Based Preparation Guide</div>
            <div class="welcome-subtitle">
                Upload your documents, ask questions, and learn smarter.
            </div>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-icon">📚</div>
                    <div class="feature-title">Ask from Documents</div>
                    <div class="feature-desc">Query specific concepts, formulas, and facts with exact page citations.</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🧠</div>
                    <div class="feature-title">Understand Topics</div>
                    <div class="feature-desc">Break down complex materials and theories into structured explanations.</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📝</div>
                    <div class="feature-title">Prepare for Exams</div>
                    <div class="feature-desc">Practice high-yield questions, test comprehension, and master key topics.</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📖</div>
                    <div class="feature-title">Summarize Documents</div>
                    <div class="feature-desc">Get instant high-level overviews and key highlights from multi-page PDFs.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Render chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📚 Sources", expanded=False):
                citations_html = "".join([
                    f'<div class="source-citation-item"><span>📄</span><span>{src["source"]}</span><span class="source-page-badge">Page {src["page"]}</span></div>'
                    for src in message["sources"]
                ])
                st.markdown(citations_html, unsafe_allow_html=True)
        if message.get("time"):
            st.markdown(f'<div class="msg-timestamp">🕒 {message["time"]}</div>', unsafe_allow_html=True)

# Persistent Quick Study Prompts (stay accessible in the chat area at all times)
if doc_count > 0:
    st.markdown(
        "<p style='font-size: 0.86rem; font-weight: 700; color: var(--text-secondary); margin-top: 1.25rem; margin-bottom: 0.5rem;'>💡 Quick Study Prompts (click anytime):</p>",
        unsafe_allow_html=True,
    )
    qp_col1, qp_col2, qp_col3 = st.columns(3)
    with qp_col1:
        if st.button("📋 Summarize Key Concepts", key="qp_summary_persistent", use_container_width=True):
            st.session_state.quick_prompt = "Provide a comprehensive summary of the key concepts and main topics from the uploaded documents."
            st.rerun()
    with qp_col2:
        if st.button("❓ Core Topics & Definitions", key="qp_topics_persistent", use_container_width=True):
            st.session_state.quick_prompt = "What are the core topics, formulas, and definitions explained in these documents?"
            st.rerun()
    with qp_col3:
        if st.button("📝 Exam Practice Questions", key="qp_exams_persistent", use_container_width=True):
            st.session_state.quick_prompt = "Generate 5 important exam preparation practice questions based on the content of these documents."
            st.rerun()

# Chat input with updated placeholder and quick starter handling
quick_question = st.session_state.pop("quick_prompt", None)
chat_input_val = st.chat_input("Ask anything about your uploaded documents...")
user_question = quick_question or chat_input_val

if user_question:
    # Display user's message immediately
    current_time = datetime.now().strftime("%H:%M:%S")
    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": user_question,
            "sources": [],
            "time": current_time,
        }
    )
    with st.chat_message("user"):
        st.markdown(user_question)
        st.markdown(f'<div class="msg-timestamp">🕒 {current_time}</div>', unsafe_allow_html=True)

    api_key = get_api_key()

    with st.chat_message("assistant"):
        if not api_key:
            error_msg = (
                "⚠️ Please provide your Groq API key in the sidebar "
                "(or set GROQ_API_KEY in your .env file) before asking questions."
            )
            st.warning(error_msg)
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                    "time": datetime.now().strftime("%H:%M:%S"),
                }
            )
        elif st.session_state.vector_store is None:
            error_msg = "⚠️ Please upload and process at least one PDF document first."
            st.warning(error_msg)
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                    "time": datetime.now().strftime("%H:%M:%S"),
                }
            )
        else:
            with st.spinner("🔍 Retrieving relevant information and generating answer..."):
                try:
                    retrieved_docs = similarity_search(
                        st.session_state.vector_store, user_question, k=top_k
                    )
                    llm = get_llm(api_key, model_name=selected_model)
                    result = answer_question(llm, retrieved_docs, user_question)

                    st.markdown(result["answer"])
                    if result["sources"]:
                        with st.expander("📚 Sources", expanded=True):
                            citations_html = "".join([
                                f'<div class="source-citation-item"><span>📄</span><span>{src["source"]}</span><span class="source-page-badge">Page {src["page"]}</span></div>'
                                for src in result["sources"]
                            ])
                            st.markdown(citations_html, unsafe_allow_html=True)

                    ans_time = datetime.now().strftime("%H:%M:%S")
                    st.markdown(f'<div class="msg-timestamp">🕒 {ans_time}</div>', unsafe_allow_html=True)

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["sources"],
                            "time": ans_time,
                        }
                    )
                except Exception as e:
                    error_msg = f"❌ An error occurred: {e}"
                    st.error(error_msg)
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": error_msg,
                            "sources": [],
                            "time": datetime.now().strftime("%H:%M:%S"),
                        }
                    )
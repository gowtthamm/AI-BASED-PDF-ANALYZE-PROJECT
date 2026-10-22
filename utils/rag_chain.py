"""
rag_chain.py
------------
Wires the FAISS retriever together with Groq LLM and a live Web Search engine
(Tavily) to answer questions using BOTH uploaded document context and live internet details.
"""

import os
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

# The default active model name for Groq.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
NO_INFO_MESSAGE = "The uploaded documents do not contain this information, and no relevant details were found online."

# System + human prompt that instructs Groq how to blend PDF data with Internet web facts.
QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an intelligent hybrid research assistant. Your goal is to provide a comprehensive, "
            "detailed, and thorough answer to the user's question.\n\n"
            "You have been provided with two sources of information:\n"
            "1. EXTRACTED TEXT FROM UPLOADED STUDY MATERIALS: Use this as your primary authoritative reference.\n"
            "2. LIVE INTERNET SEARCH RESULTS: Use this to enrich the answer, provide real-world context, or answer the question if the uploaded document lacks details.\n\n"
            "Execution Instructions:\n"
            "- Carefully blend the information from the uploaded document context and the web search facts into a single cohesive, educational response.\n"
            "- Clearly mention or differentiate what information came from the 'Uploaded Document' vs what was added via 'Web Search' context.\n"
            "- Use clean markdown layout, paragraphs, and bullet points to make it easy to read.\n"
            f"- If absolutely no information is available in either the document context or the internet results, respond with exactly: \"{NO_INFO_MESSAGE}\"\n\n"
            "Uploaded Document Context:\n{document_context}\n\n"
            "Live Internet Search Context:\n{internet_context}",
        ),
        ("human", "{question}"),
    ]
)

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that writes concise summaries of "
            "uploaded study and reference documents. Summarize the key points "
            "in clear bullet points using ONLY the content provided.",
        ),
        ("human", "Document content:\n\n{context}\n\nWrite a concise summary."),
    ]
)


def get_llm(api_key: str, temperature: float = 0.4, model_name: Optional[str] = None) -> ChatGroq:
    """Instantiate the Groq chat model."""
    chosen_model = model_name or os.getenv("GROQ_MODEL_NAME") or DEFAULT_GROQ_MODEL
    return ChatGroq(
        model=chosen_model,
        groq_api_key=api_key,
        temperature=temperature,
    )


def format_context(documents: List[Document]) -> str:
    """Concatenate retrieved chunks into a single context string."""
    parts = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Chunk {i} | Source: {source}, Page: {page}]\n{doc.page_content}")
    return "\n\n".join(parts)


def build_sources(documents: List[Document]) -> List[Dict]:
    """Build a list of document citations."""
    seen = set()
    sources = []
    for doc in documents:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        key = (source, page)
        if key not in seen:
            seen.add(key)
            sources.append({"source": source, "page": page})
    return sources


def answer_question(
    llm: ChatGroq, retrieved_docs: List[Document], question: str
) -> Dict:
    """
    Executes a dual-lookup strategy: searches local FAISS chunks and queries
    the live internet concurrently, combining the findings via Groq.
    """
    # 1. Gather document text
    doc_context = format_context(retrieved_docs) if retrieved_docs else "No document text available."

    # 2. Gather live internet text using Tavily
    internet_context = "No internet results retrieved."
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    
    if tavily_api_key:
        try:
            os.environ.setdefault("TAVILY_API_KEY", tavily_api_key)
            search_tool = TavilySearch(max_results=3)
            search_results = search_tool.invoke(question)
            # New langchain_tavily returns a formatted string directly
            internet_context = search_results if isinstance(search_results, str) else str(search_results)
        except Exception as e:
            internet_context = f"Web search tool encountered an operational delay: {e}"
    else:
        internet_context = "[Notice: Web search skipped because TAVILY_API_KEY is missing from .env]"

    # 3. Stream data through the LLM execution pipeline
    chain = QA_PROMPT | llm | StrOutputParser()

    try:
        answer = chain.invoke({
            "document_context": doc_context, 
            "internet_context": internet_context, 
            "question": question
        })
    except Exception as exc:
        raise RuntimeError(f"Groq API call failed: {exc}") from exc

    # Determine citations
    sources = [] if NO_INFO_MESSAGE in answer else build_sources(retrieved_docs)

    return {"answer": answer.strip(), "sources": sources}


def summarize_documents(llm: ChatGroq, documents: List[Document]) -> str:
    """Generate a concise summary of indexed documents."""
    if not documents:
        return "No documents available to summarize."

    max_chunks = 40
    sample = documents[:max_chunks]
    context = format_context(sample)

    chain = SUMMARY_PROMPT | llm | StrOutputParser()
    try:
        summary = chain.invoke({"context": context})
    except Exception as exc:
        raise RuntimeError(f"Groq API call failed during summarization: {exc}") from exc

    return summary.strip()
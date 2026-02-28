from pathlib import Path
from typing import Dict, List, Tuple

import chromadb
from chromadb.utils import embedding_functions
import ollama


DB_DIR = Path("index/chroma")
COLLECTION_NAME = "carbon_docs"

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"

# Local chat model. You can switch to "mistral:7b" for faster runtime.
CHAT_MODEL = "mistral:7b"


def get_collection() -> chromadb.api.models.Collection.Collection:
    """
    Load the persistent Chroma collection using local Ollama embeddings.
    """
    client = chromadb.PersistentClient(path=str(DB_DIR))

    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        model_name=EMBED_MODEL,
        url=OLLAMA_URL,
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def build_context(docs: List[str], metas: List[dict], max_chars: int = 5000) -> Tuple[str, List[str]]:
    """
    Build a context string from retrieved chunks and return the list of unique sources.
    """
    sources: List[str] = []
    parts: List[str] = []

    for text, meta in zip(docs, metas):
        src = meta.get("source", "unknown")
        if src not in sources:
            sources.append(src)

        parts.append(f"[Source: {src}]\n{text}")

    ctx = "\n\n---\n\n".join(parts)
    if len(ctx) > max_chars:
        ctx = ctx[:max_chars] + "\n\n[Context truncated due to length.]"

    return ctx, sources

def retrieval_strength(distances: list[float]) -> tuple[str, float]:
    """
    Returns (label, score) where label is High/Medium/Low and score is a confidence-like value in [0, 1].
    Heuristic for cosine distance: lower distance => higher confidence.
    """
    if not distances:
        return "Low", 0.0

    best = float(min(distances))  # smaller is better

    # Map distance to a 0-1 score (simple piecewise heuristic)
    if best <= 0.25:
        return "High", 0.9
    if best <= 0.45:
        return "Medium", 0.6
    return "Low", 0.3


def is_weak_retrieval(distances: list[float], threshold: float = 0.45) -> bool:
    """Weak retrieval if the best match is still too far."""
    if not distances:
        return True
    return min(distances) > threshold

def answer(question: str, k: int = 4) -> Dict:
    """
    Answer a question using RAG:
    - Retrieve top-k chunks from the vector DB
    - Ask local LLM to answer using ONLY the retrieved context
    """
    if not DB_DIR.exists():
        raise RuntimeError("Index not found. Run `python app/ingest.py` first.")

    col = get_collection()
    res = col.query(query_texts=[question], n_results=k, include=["documents", "metadatas", "distances"])

    docs = res["documents"][0] if res.get("documents") else []

    if not docs:
        return {
            "answer": "I don't know based on the provided documents.",
            "sources": [],
            "distances": [],
            "retrieved_chunks": [],
            "retrieval_strength": "Low",
            "confidence": 0.0,
        }

    metas = res["metadatas"][0] if res.get("metadatas") else []
    distances = res["distances"][0] if res.get("distances") else []

    strength, conf = retrieval_strength(distances)

    # If retrieval is weak, do NOT call the LLM (prevents hallucination + speeds up)
    if is_weak_retrieval(distances, threshold=0.45):
        sources = list({m.get("source", "unknown") for m in metas}) if metas else []
        return {
            "answer": (
                "I don't know based on the provided documents. "
                "The retrieved context seems insufficient to answer reliably."
            ),
            "sources": sources,
            "distances": distances,
            "retrieved_chunks": [
                {"source": m.get("source", "unknown"), "text": d}
                for d, m in zip(docs, metas)
            ],
            "retrieval_strength": strength,
            "confidence": conf,
        }


    context, sources = build_context(docs, metas)

    system = (
        "You are a Carbon Credit Risk Analyst Assistant.\n"
        "You evaluate carbon credit concepts and projects using provided documents.\n"
        "\n"
        "STRICT RULES:\n"
        "1) Use ONLY the provided context.\n"
        "2) If information is missing, say: \"I don't know based on the provided documents.\".\n"
        "3) Do NOT use external knowledge.\n"
        "4) Be precise, technical, and concise.\n"
        "5) Always cite sources from the context.\n"
        "\n"
        "RESPONSE STRUCTURE:\n"
        "## Summary\n"
        "Short explanation answering the question.\n"
        "\n"
        "## Key Points\n"
        "- Bullet points with main technical facts.\n"
        "\n"
        "## Risk & Quality Implications\n"
        "Discuss implications for carbon credit quality considering:\n"
        "- Additionality\n"
        "- Baseline credibility\n"
        "- MRV robustness\n"
        "- Leakage risk\n"
        "- Permanence risk\n"
        "\n"
        "## Red Flags (if any)\n"
        "List potential quality concerns or uncertainties.\n"
        "\n"
        "## Sources\n"
        "List the source document filenames used.\n"
        "Do not mention missing context or documents.\n"
    )

    user = f"Context:\n{context}\n\nQuestion:\n{question}"

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": 0.0},
    )

    return {
        "answer": response["message"]["content"],
        "sources": sources,
        "distances": distances,
        "retrieved_chunks": [
            {"source": m.get("source", "unknown"), "text": d}
            for d, m in zip(docs, metas)
        ],
        "retrieval_strength": strength,
        "confidence": conf,
        "best_distance": min(distances) if distances else None,
    }

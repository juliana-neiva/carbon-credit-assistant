import glob
import os
from pathlib import Path
from typing import List, Tuple

import chromadb
from chromadb.utils import embedding_functions


DOCS_DIR = Path("data/docs")
DB_DIR = Path("index/chroma")
COLLECTION_NAME = "carbon_docs"

# Ollama local embeddings model (no API keys required)
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """
    Split text into overlapping chunks.

    The overlap helps preserve context across chunk boundaries and generally improves retrieval quality.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: List[str] = []
    i = 0
    n = len(text)

    while i < n:
        chunks.append(text[i : i + chunk_size])
        i += chunk_size - overlap

    return chunks


def load_documents(doc_dir: Path) -> List[Tuple[str, str]]:
    """
    Load .md/.txt documents from the given directory (recursive).
    Returns list of (filepath, content).
    """
    patterns = [
        str(doc_dir / "**" / "*.md"),
        str(doc_dir / "**" / "*.txt"),
    ]

    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))

    docs: List[Tuple[str, str]] = []
    for fp in sorted(set(files)):
        with open(fp, "r", encoding="utf-8") as f:
            docs.append((fp, f.read()))

    return docs


def get_collection() -> chromadb.api.models.Collection.Collection:
    """
    Create/load a persistent Chroma collection with Ollama embeddings (local).
    """
    DB_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(DB_DIR))

    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        model_name=EMBED_MODEL,
        url=OLLAMA_URL,
    )

    # cosine space tends to work well for text embeddings
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )

    return collection


def clear_collection(collection: chromadb.api.models.Collection.Collection) -> None:
    """
    Remove all items from the collection.
    Chroma doesn't have a simple truncate; we delete by fetching all ids.
    """
    current = collection.get(include=["metadatas"])
    ids = current.get("ids", [])
    if ids:
        collection.delete(ids=ids)


def run_ingest(
    doc_dir: Path = DOCS_DIR,
    chunk_size: int = 900,
    overlap: int = 150,
    reset_index: bool = False,
) -> None:
    """
    Ingest documents into the persistent vector DB.

    - Reads docs
    - Chunks text
    - Creates embeddings locally (Ollama)
    - Upserts into Chroma
    """
    if not doc_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {doc_dir}")

    collection = get_collection()

    if reset_index:
        clear_collection(collection)

    docs = load_documents(doc_dir)
    if not docs:
        raise RuntimeError(f"No .md/.txt documents found under: {doc_dir}")

    ids: List[str] = []
    texts: List[str] = []
    metas: List[dict] = []

    for fp, content in docs:
        source = os.path.basename(fp)
        chunks = chunk_text(content, chunk_size=chunk_size, overlap=overlap)

        for j, ch in enumerate(chunks):
            # Stable ID so repeated ingestion overwrites, not duplicates
            doc_id = f"{source}::chunk{j:04d}"
            ids.append(doc_id)
            texts.append(ch)
            metas.append({"source": source})

    collection.upsert(ids=ids, documents=texts, metadatas=metas)


if __name__ == "__main__":
    # CLI usage:
    # python app/ingest.py
    # python app/ingest.py --reset
    import argparse

    parser = argparse.ArgumentParser(description="Ingest local docs into Chroma (with Ollama embeddings).")
    parser.add_argument("--reset", action="store_true", help="Clear and rebuild the index.")
    parser.add_argument("--chunk_size", type=int, default=900)
    parser.add_argument("--overlap", type=int, default=150)
    args = parser.parse_args()

    run_ingest(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        reset_index=args.reset,
    )
    print("Ingestion complete.")
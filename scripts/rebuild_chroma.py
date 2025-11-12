"""Rebuild the Chroma vector index with one document per book.

Usage:
    python scripts/rebuild_chroma.py

This script expects an OPENAI_API_KEY to be available (via .env or env var).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
try:
    from langchain.schema import Document
except ModuleNotFoundError:  # langchain >= 0.2.0
    from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def build_documents(books_df: pd.DataFrame) -> list[Document]:
    """Create one LangChain Document per book row."""

    documents: list[Document] = []

    for _, row in books_df.iterrows():
        title = row.get("title", "Unknown title")
        authors = row.get("authors") or "Unknown author"
        description = row.get("description") or ""
        simple_category = row.get("simple_category") or row.get("simple_categories")

        page_parts = [
            f"Title: {title}",
            f"Authors: {authors}",
        ]

        if isinstance(simple_category, str) and simple_category.strip():
            page_parts.append(f"Category: {simple_category}")

        if isinstance(description, str) and description.strip():
            page_parts.append("Description: " + description.strip())

        page_content = "\n".join(page_parts)

        metadata = {
            "isbn13": str(row.get("isbn13")) if not pd.isna(row.get("isbn13")) else None,
            "isbn10": str(row.get("isbn10")) if not pd.isna(row.get("isbn10")) else None,
            "title": title,
            "authors": authors,
            "simple_category": simple_category,
        }

        documents.append(Document(page_content=page_content, metadata=metadata))

    return documents


def main() -> None:
    load_dotenv()

    data_path = Path("data/books_with_emotions.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    books = pd.read_csv(data_path)

    # Drop rows without ISBN or description to avoid empty docs
    books = books.dropna(subset=["isbn13", "description"]).copy()
    books["isbn13"] = books["isbn13"].astype(np.int64).astype(str)

    documents = build_documents(books)

    persist_dir = Path("data/chroma_index")
    if persist_dir.exists():
        shutil.rmtree(persist_dir)

    persist_dir.mkdir(parents=True, exist_ok=True)

    embeddings = OpenAIEmbeddings()
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name="books",
    )

    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_name="books",
    )
    doc_count = vectorstore._collection.count() if hasattr(vectorstore, "_collection") else len(documents)

    print(f"✓ Rebuilt Chroma index with {doc_count} documents")
    print(f"✓ Persisted to {persist_dir}")


if __name__ == "__main__":
    main()

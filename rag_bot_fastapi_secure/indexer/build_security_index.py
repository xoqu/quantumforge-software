"""Build the isolated Chroma collection inside Docker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
import yaml
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_COLLECTION = "neon_rift_security_test"
PROTECTED_COLLECTION = "neon_rift_knowledge"
DEFAULT_MODEL = "BAAI/bge-m3"
FRONTMATTER_PATTERN = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the isolated security index.")
    parser.add_argument("--knowledge-base", type=Path, default=Path("/knowledge_base"))
    parser.add_argument("--malicious-dir", type=Path, default=Path("/malicious_docs"))
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--chroma-host",
        default=os.getenv("CHROMA_HOST", "chroma"),
    )
    parser.add_argument(
        "--chroma-port",
        type=int,
        default=int(os.getenv("CHROMA_PORT", "8000")),
    )
    parser.add_argument("--device", default=os.getenv("DEVICE", "cpu"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size-words", type=int, default=200)
    parser.add_argument("--chunk-overlap-words", type=int, default=40)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("/reports/security_index_report.json"),
    )
    parser.add_argument("--recreate", action="store_true")
    return parser.parse_args()


def parse_markdown(path: Path) -> tuple[dict[str, Any], str]:
    raw_text = path.read_text(encoding="utf-8")
    match = FRONTMATTER_PATTERN.match(raw_text)
    if not match:
        return {}, raw_text.strip()
    metadata = yaml.safe_load(match.group(1)) or {}
    return metadata, raw_text[match.end() :].strip()


def read_normal_document(path: Path, root: Path) -> Document:
    frontmatter, body = parse_markdown(path)
    if not body:
        raise ValueError(f"Document contains no text: {path}")
    relative_path = path.relative_to(root).as_posix()
    return Document(
        page_content=body,
        metadata={
            "source": f"knowledge_base/{relative_path}",
            "title": str(frontmatter.get("title", path.stem)),
            "category": str(frontmatter.get("category", "unknown")),
            "source_id": str(
                frontmatter.get("source_id", path.stem.split("_", 1)[0])
            ),
            "security_test": False,
        },
    )


def read_malicious_document(path: Path, root: Path, index: int) -> Document:
    body = path.read_text(encoding="utf-8").strip()
    if not body:
        raise ValueError(f"Malicious test document contains no text: {path}")
    return Document(
        page_content=body,
        metadata={
            "source": f"malicious_docs/{path.relative_to(root).as_posix()}",
            "title": f"Security test document {index}",
            "category": "security_test",
            "source_id": f"attack-{index:03d}",
            "security_test": True,
        },
    )


def load_documents(
    knowledge_base: Path,
    malicious_dir: Path,
) -> tuple[list[Document], int]:
    if not knowledge_base.is_dir():
        raise FileNotFoundError(f"Knowledge base does not exist: {knowledge_base}")
    if not malicious_dir.is_dir():
        raise FileNotFoundError(f"Malicious directory does not exist: {malicious_dir}")

    normal_paths = sorted(
        path
        for path in knowledge_base.iterdir()
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}
    )
    malicious_paths = sorted(
        path
        for path in malicious_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}
    )
    if not normal_paths:
        raise FileNotFoundError(
            f"No .md or .txt documents found in {knowledge_base}"
        )
    if not malicious_paths:
        raise FileNotFoundError(f"No attack documents found in {malicious_dir}")

    documents = [
        read_normal_document(path, knowledge_base) for path in normal_paths
    ]
    documents.extend(
        read_malicious_document(path, malicious_dir, index)
        for index, path in enumerate(malicious_paths, start=1)
    )
    return documents, len(malicious_paths)


def split_documents(
    documents: list[Document],
    chunk_size_words: int,
    chunk_overlap_words: int,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_words,
        chunk_overlap=chunk_overlap_words,
        length_function=lambda text: len(text.split()),
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        add_start_index=True,
        strip_whitespace=True,
    )
    chunks = splitter.split_documents(documents)
    sequence_by_source: defaultdict[str, int] = defaultdict(int)

    for chunk in chunks:
        source = str(chunk.metadata["source"])
        chunk_number = sequence_by_source[source]
        sequence_by_source[source] += 1
        chunk.metadata["chunk_id"] = f"{chunk.metadata['source_id']}:{chunk_number:04d}"
        chunk.metadata["chunk_number"] = chunk_number
        chunk.metadata["word_count"] = len(chunk.page_content.split())
    return chunks


def deterministic_chunk_id(chunk: Document) -> str:
    identity = (
        f"{chunk.metadata['source']}|{chunk.metadata['chunk_id']}|{chunk.page_content}"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def connect_to_chroma(host: str, port: int) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, 31):
        try:
            client = chromadb.HttpClient(host=host, port=port)
            client.heartbeat()
            return client
        except Exception as error:
            last_error = error
            print(f"Waiting for ChromaDB ({attempt}/30): {error}")
            time.sleep(2)
    raise RuntimeError("ChromaDB did not become available") from last_error


def collection_exists(client: Any, name: str) -> bool:
    return any(collection.name == name for collection in client.list_collections())


def main() -> None:
    args = parse_args()
    if args.collection == PROTECTED_COLLECTION:
        raise SystemExit(f"Refusing to modify {PROTECTED_COLLECTION}")
    if args.chunk_size_words <= 0:
        raise ValueError("--chunk-size-words must be positive")
    if not 0 <= args.chunk_overlap_words < args.chunk_size_words:
        raise ValueError("Chunk overlap must be smaller than chunk size")

    started_at = time.perf_counter()
    documents, malicious_document_count = load_documents(
        args.knowledge_base,
        args.malicious_dir,
    )
    chunks = split_documents(
        documents,
        args.chunk_size_words,
        args.chunk_overlap_words,
    )

    client = connect_to_chroma(args.chroma_host, args.chroma_port)
    if collection_exists(client, args.collection):
        if not args.recreate:
            raise SystemExit(
                f"Collection {args.collection!r} exists; use --recreate"
            )
        client.delete_collection(args.collection)

    embeddings = HuggingFaceEmbeddings(
        model_name=args.model,
        model_kwargs={"device": args.device},
        encode_kwargs={"batch_size": args.batch_size, "normalize_embeddings": True},
    )
    embedding_dimension = len(embeddings.embed_query("dimension probe"))
    vector_store = Chroma(
        client=client,
        collection_name=args.collection,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )
    vector_store.add_documents(
        documents=chunks,
        ids=[deterministic_chunk_id(chunk) for chunk in chunks],
    )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_model": args.model,
        "embedding_dimension": embedding_dimension,
        "collection": args.collection,
        "normal_document_count": len(documents) - malicious_document_count,
        "malicious_document_count": malicious_document_count,
        "chunk_count": len(chunks),
        "malicious_chunk_count": sum(
            bool(chunk.metadata.get("security_test")) for chunk in chunks
        ),
        "indexed_count": client.get_collection(args.collection).count(),
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

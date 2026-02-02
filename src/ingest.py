"""Data ingestion module for downloading and chunking research papers."""

import json
import os
import time
from pathlib import Path

import arxiv
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from pypdf import PdfReader

from src.config import PAPERS_DIR, METADATA_FILE


def download_papers(
    query: str,
    max_results: int = 5,
    output_dir: str = PAPERS_DIR,
    max_retries: int = 3,
) -> list[dict]:
    """
    Download papers from arxiv matching the query.

    Args:
        query: Search query for arxiv
        max_results: Maximum number of papers to download
        output_dir: Directory to save PDFs
        max_retries: Number of retries on rate limiting

    Returns:
        List of metadata dicts for downloaded papers
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers = []
    for attempt in range(max_retries):
        try:
            for result in client.results(search):
                paper_id = result.entry_id.split("/")[-1]
                pdf_path = os.path.join(output_dir, f"{paper_id}.pdf")

                if not os.path.exists(pdf_path):
                    result.download_pdf(dirpath=output_dir, filename=f"{paper_id}.pdf")

                metadata = {
                    "id": paper_id,
                    "title": result.title,
                    "authors": [str(a) for a in result.authors],
                    "abstract": result.summary,
                    "published": str(result.published),
                    "pdf_path": pdf_path,
                    "url": result.entry_id,
                }
                papers.append(metadata)
            break
        except Exception as e:
            if "rate" in str(e).lower() and attempt < max_retries - 1:
                print(
                    f"Rate limited, sleeping 60s (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(60)
            else:
                raise

    metadata_path = Path(METADATA_FILE)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(papers, f, indent=2)

    return papers


def chunk_papers(
    metadata_path: str = METADATA_FILE,
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
) -> list[Document]:
    """
    Chunk papers into LlamaIndex Document objects.

    Args:
        metadata_path: Path to metadata.json file
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks

    Returns:
        List of LlamaIndex Document objects with metadata
    """
    with open(metadata_path) as f:
        papers_metadata = json.load(f)

    documents = []
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    for paper in papers_metadata:
        pdf_path = paper["pdf_path"]

        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            print(f"Failed to parse PDF {pdf_path}: {e}")
            continue

        if not text.strip():
            print(f"No text extracted from {pdf_path}")
            continue

        doc = Document(
            text=text,
            metadata={
                "paper_id": paper["id"],
                "title": paper["title"],
                "authors": ", ".join(paper["authors"]),
                "published": paper["published"],
                "url": paper["url"],
            },
        )

        chunks = splitter.get_nodes_from_documents([doc])
        for i, chunk in enumerate(chunks):
            chunk_doc = Document(
                text=chunk.get_content(),
                metadata={
                    **doc.metadata,
                    "chunk_index": i,
                },
            )
            documents.append(chunk_doc)

    return documents


if __name__ == "__main__":
    papers = download_papers("retrieval augmented generation", max_results=5)
    print(f"Downloaded {len(papers)} papers")

    docs = chunk_papers()
    print(f"Created {len(docs)} document chunks")

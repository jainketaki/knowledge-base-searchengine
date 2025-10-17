"""
Core RAG engine: PDF extraction, chunking, embedding, search, summarization.
Designed for offline use with local models saved under `models/` (optional).
"""
from typing import List, Tuple
import os
import re

from PyPDF2 import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity


def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        try:
            txt = page.extract_text()
        except Exception:
            txt = None
        if txt:
            texts.append(txt)
    return "\n".join(texts)


def chunk_text(text: str, words_per_chunk: int = 500) -> List[str]:
    # Normalize whitespace and split by words
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split(" ")
    chunks = []
    for i in range(0, len(words), words_per_chunk):
        chunk = " ".join(words[i:i+words_per_chunk])
        chunks.append(chunk)
    return chunks


class RAGEngine:
    def __init__(self,
                 embed_model_name: str = "all-MiniLM-L6-v2",
                 summarizer_model_name: str = "facebook/bart-large-cnn",
                 local_models_dir: str = "models"):
        self.local_models_dir = local_models_dir
        self.embed_model_name = embed_model_name
        self.summarizer_model_name = summarizer_model_name

        # Load or point to local sentence-transformers model
        embed_path = self._local_or_remote_path(embed_model_name)
        print(f"Loading embedding model from {embed_path} ...")
        self.embedder = SentenceTransformer(embed_path)

        # Prepare summarizer pipeline
        summarizer_path = self._local_or_remote_path(summarizer_model_name)
        print(f"Loading summarization model from {summarizer_path} ...")
        # Use pipeline with local model directory; tokenizer/model auto-detect
        self.summarizer = pipeline("summarization", model=summarizer_path, device=-1)

        # In-memory storage
        self.chunks: List[str] = []
        self.embeddings: List[np.ndarray] = []

    def _local_or_remote_path(self, name: str) -> str:
        # If a local folder under models exists, prefer it
        local_path = os.path.join(self.local_models_dir, name)
        if os.path.isdir(local_path):
            return local_path
        return name

    def ingest_pdf(self, pdf_path: str) -> int:
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            return 0
        chunks = chunk_text(text, words_per_chunk=500)
        self.chunks = chunks
        # Compute embeddings
        embeddings = self.embedder.encode(chunks, convert_to_numpy=True)
        self.embeddings = embeddings
        return len(chunks)

    def query(self, question: str, top_k: int = 3) -> Tuple[str, List[Tuple[int, float, str]]]:
        if not self.chunks or len(self.chunks) == 0:
            raise ValueError("No document ingested. Please upload a PDF first.")
        q_emb = self.embedder.encode([question], convert_to_numpy=True)
        sims = cosine_similarity(q_emb, self.embeddings)[0]
        # Get top k indices
        idxs = np.argsort(sims)[::-1][:top_k]
        top_chunks = []
        for idx in idxs:
            top_chunks.append((int(idx), float(sims[idx]), self.chunks[idx]))

        # Combine top chunks into context
        context = "\n\n".join([c for (_, _, c) in top_chunks])
        # Summarize context
        summary = self.summarizer(context, max_length=180, min_length=50, do_sample=False)[0]["summary_text"]
        return summary, top_chunks


if __name__ == "__main__":
    print("rag_engine.py: basic smoke test")
    # This basic test only checks imports and class creation (no models downloaded here)
    try:
        engine = RAGEngine()
        print("RAGEngine created (models may attempt to download if not present)." )
    except Exception as e:
        print("Error creating RAGEngine:", e)

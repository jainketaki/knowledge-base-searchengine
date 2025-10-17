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


try:
    from keybert import KeyBERT
    _HAS_KEYBERT = True
except ImportError:
    _HAS_KEYBERT = False


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


def token_chunk_text(text: str, tokenizer, tokens_per_chunk: int = 512, overlap: int = 64) -> list:
    # Tokenize the text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + tokens_per_chunk, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        if end == len(tokens):
            break
        start += tokens_per_chunk - overlap
    return chunks


def sentence_chunk_text(text: str, max_sentences: int = 5) -> list:
    # Split text into sentences using regex (simple, language-agnostic)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = ' '.join(sentences[i:i+max_sentences]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def paragraph_chunk_text(text: str, max_paragraphs: int = 2) -> list:
    # Split text into paragraphs (double newline or similar)
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    for i in range(0, len(paragraphs), max_paragraphs):
        chunk = '\n'.join(paragraphs[i:i+max_paragraphs]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


class RAGEngine:
    def __init__(self,
                 embed_model_name: str = "all-MiniLM-L6-v2",
                 # Use a smaller CPU-friendly summarizer by default for faster local runs
                 summarizer_model_name: str = "sshleifer/distilbart-cnn-12-6",
                 local_models_dir: str = "models",
                 chunking_mode: str = "token",
                 tokens_per_chunk: int = 512,
                 overlap: int = 64,
                 max_sentences: int = 5,
                 max_paragraphs: int = 2):
        self.local_models_dir = local_models_dir
        self.embed_model_name = embed_model_name
        self.summarizer_model_name = summarizer_model_name
        self.chunking_mode = chunking_mode
        self.tokens_per_chunk = tokens_per_chunk
        self.overlap = overlap
        self.max_sentences = max_sentences
        self.max_paragraphs = max_paragraphs

        # Load or point to local sentence-transformers model
        embed_path = self._local_or_remote_path(embed_model_name)
        print(f"Loading embedding model from {embed_path} ...")
        self.embedder = SentenceTransformer(embed_path)

        # Prepare summarizer pipeline with explicit CPU config
        summarizer_path = self._local_or_remote_path(summarizer_model_name)
        print(f"Loading summarization model from {summarizer_path} ...")
        # Load model and tokenizer explicitly to control device placement
        model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_path).cpu()
        tokenizer = AutoTokenizer.from_pretrained(summarizer_path)
        self.summarizer = pipeline("summarization", 
                                 model=model, 
                                 tokenizer=tokenizer,
                                 device="cpu",  # Explicit CPU
                                 framework="pt")  # Use PyTorch backend
        self.tokenizer = tokenizer

        # In-memory storage
        self.chunks: list = []
        self.embeddings: list = []

    def _local_or_remote_path(self, name: str) -> str:
        # If a local folder under models exists, prefer it
        local_path = os.path.join(self.local_models_dir, name)
        if os.path.isdir(local_path):
            return local_path
        return name

    def _clean_text(self, text: str) -> str:
        # Replace multiple spaces/newlines with a single space, fix spacing after punctuation
        text = re.sub(r'[ \t\r\f\v]+', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)  # Ensure space after punctuation
        return text.strip()

    def ingest_pdf(self, pdf_path: str) -> int:
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            return 0
        if self.chunking_mode == "token":
            chunks = token_chunk_text(text, self.tokenizer, self.tokens_per_chunk, self.overlap)
        elif self.chunking_mode == "sentence":
            chunks = sentence_chunk_text(text, self.max_sentences)
        elif self.chunking_mode == "paragraph":
            chunks = paragraph_chunk_text(text, self.max_paragraphs)
        else:
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
        # Prepend the question as a prompt for the summarizer
        prompt = f"Question: {question}\nContext: {context}"
        # Truncate context to a reasonable size for CPU summarizers
        if len(prompt) > 4000:
            prompt = prompt[:4000]
        # Summarize context with safe parameters for CPU
        try:
            result = self.summarizer(prompt,
                                   max_length=180,
                                   min_length=50,
                                   do_sample=False,
                                   num_beams=1,
                                   temperature=1.0)
            summary = result[0]["summary_text"].strip()
            summary = self._clean_text(summary)
        except Exception as e:
            print(f"Summarization error: {str(e)}")
            words = context.split()[:180]
            summary = " ".join(words) + "..."
        # Fallback: if summary is too generic or empty, return the most relevant chunk
        if not summary or len(summary.split()) < 5 or summary.lower().startswith("question: "):
            summary = self._clean_text(top_chunks[0][2][:500] + ("..." if len(top_chunks[0][2]) > 500 else ""))
        return summary, top_chunks

    def extract_topics(self, top_n: int = 5) -> list:
        if not self.chunks or len(self.chunks) == 0:
            return []
        context = " ".join(self.chunks)
        # Prefer KeyBERT if available
        if _HAS_KEYBERT:
            kw_model = KeyBERT(model="all-MiniLM-L6-v2")
            keywords = kw_model.extract_keywords(context, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=top_n)
            return [kw for kw, _ in keywords]
        # Fallback: use LLM summarizer prompt
        prompt = f"List the {top_n} most important topics or concepts in this document as bullet points.\n{context[:4000]}"
        result = self.summarizer(prompt, max_length=80, min_length=20, do_sample=False)
        topics = result[0]["summary_text"].strip()
        return topics.split("\n")

    def generate_quiz(self, num_questions: int = 3) -> list:
        if not self.chunks or len(self.chunks) == 0:
            return []
        context = " ".join(self.chunks)
        # Limit context to avoid model truncation
        context = context[:3000]
        prompt = (
            f"Create {num_questions} quiz questions and answers from the following document. "
            "Format as: Q1: <question>\nA1: <answer>\nQ2: <question>\nA2: <answer>\nQ3: <question>\nA3: <answer>\n" + context
        )
        result = self.summarizer(prompt, max_length=220, min_length=60, do_sample=False)
        qa_text = result[0]["summary_text"].strip()
        # Parse Q/A pairs
        qa_pairs = []
        for match in re.finditer(r"Q\d+:\s*(.*?)\s*A\d+:\s*(.*?)(?=Q\d+:|$)", qa_text, re.DOTALL):
            q, a = match.groups()
            q = self._clean_text(q)
            a = self._clean_text(a)
            if q and a:
                qa_pairs.append((q, a))
        # Fallback: show raw output if parsing fails
        if not qa_pairs:
            qa_pairs = [("Quiz Output", self._clean_text(qa_text))]
        return qa_pairs


if __name__ == "__main__":
    print("rag_engine.py: basic smoke test")
    # This basic test only checks imports and class creation (no models downloaded here)
    try:
        engine = RAGEngine()
        print("RAGEngine created (models may attempt to download if not present)." )
    except Exception as e:
        print("Error creating RAGEngine:", e)

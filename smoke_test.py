"""
Simple smoke test for RAGEngine without Streamlit.
Creates a tiny PDF if none exists, ingests it, and runs a sample query.
"""
from pathlib import Path
from rag_engine import RAGEngine
from fpdf import FPDF

sample_pdf = Path('sample.pdf')

if not sample_pdf.exists():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', size=12)
    text = """
    This is a small sample document used for testing the offline RAG engine.
    The document describes a fictional project: a knowledge-base search engine.
    It includes sections about architecture, local embeddings, and summarization.
    """
    pdf.multi_cell(0, 5, text)
    pdf.output(str(sample_pdf))
    print('Created sample.pdf')

engine = RAGEngine()
count = engine.ingest_pdf(str(sample_pdf))
print('Chunks:', count)
summary, top_chunks = engine.query('What is this document about?')
print('Summary:\n', summary)
for idx, score, chunk in top_chunks:
    print('---')
    print('chunk idx:', idx, 'score:', score)
    print(chunk[:300])

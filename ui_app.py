"""
Streamlit UI for Offline RAG Search Engine
"""
import streamlit as st
from pathlib import Path
from rag_engine import RAGEngine

st.set_page_config(page_title="Offline Knowledge-base Search Engine", layout="centered")

st.title("Offline Knowledge-base Search Engine")
st.write("Upload a single PDF, ask a question, and get a summarized answer from the document (offline).")

# Sidebar for model path override
with st.sidebar:
    st.header("Model options")
    models_dir = st.text_input("Local models directory (optional)", value="models")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
question = st.text_input("Enter your question")

engine_placeholder = st.empty()

if 'engine' not in st.session_state:
    try:
        st.session_state['engine'] = RAGEngine(local_models_dir=models_dir)
    except Exception as e:
        st.error(f"Error initializing models: {e}")

if uploaded_file is not None:
    pdf_path = Path("uploaded.pdf")
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("PDF uploaded. Extracting text and building embeddings (may take a moment).")
    with st.spinner("Processing PDF..."):
        try:
            count = st.session_state['engine'].ingest_pdf(str(pdf_path))
            if count == 0:
                st.warning("No readable text found in the PDF. Make sure it's not scanned images.")
            else:
                st.info(f"Document split into {count} chunks.")
        except Exception as e:
            st.error(f"Error processing PDF: {e}")

if st.button("Generate Answer"):
    if uploaded_file is None:
        st.error("Please upload a PDF first.")
    elif not question.strip():
        st.error("Please type a question.")
    else:
        with st.spinner("Searching and generating answer..."):
            try:
                summary, top_chunks = st.session_state['engine'].query(question, top_k=3)
                st.subheader("Answer")
                st.write(summary)

                st.subheader("Top source snippets")
                for i, (idx, score, chunk) in enumerate(top_chunks, start=1):
                    st.markdown(f"**{i}. (score: {score:.4f})**")
                    st.write(chunk[:1000])
            except Exception as e:
                st.error(f"Error generating answer: {e}")

st.markdown("---")
st.caption("All processing is local. Models are loaded from the `models/` directory if present; otherwise the library may try to download them the first time.")

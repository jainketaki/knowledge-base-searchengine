"""
Streamlit UI for Offline RAG Search Engine
"""
import streamlit as st
from pathlib import Path
from rag_engine import RAGEngine

# --- Custom CSS for colors and layout ---
st.markdown(
    '''<style>
    .main-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .studybud-logo {
        font-size: 2.2rem;
        font-weight: bold;
        color: #808000;
        letter-spacing: 2px;
        font-family: 'Segoe UI', sans-serif;
        margin-right: 0.5rem;
    }
    .studybud-sub {
        font-size: 1.2rem;
        color: #222;
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp {
        background: #FAF9F6;
        color: #111 !important;
    }
    .stMarkdown, .stTextInput, .stFileUploader, .stAlert, .stSpinner, .stCaption, .stDataFrame, .stTable, .stTextArea, .stSelectbox, .stRadio, .stCheckbox, .stSlider, .stNumberInput, .stDateInput, .stTimeInput, .stColorPicker, .stButton, .stDownloadButton {
        color: #111 !important;
    }
    .stButton>button, .stDownloadButton>button {
        background-color: #808000 !important;
        color: #fff !important;
        border-radius: 6px !important;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        border: none !important;
    }
    .stTextInput>div>input, .stFileUploader>div>div {
        background: #f8f8e7;
        border-radius: 6px;
        color: #111 !important;
    }
    .stAlert {
        border-radius: 6px;
    }
    .stSpinner {
        color: #808000 !important;
    }
    .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #808000;
    }
    </style>''',
    unsafe_allow_html=True
)

# --- Header with logo and name ---
st.markdown(
    '''<div class="main-header">
        <span class="studybud-logo">📚 Study Bud</span>
        <span class="studybud-sub">Offline Knowledge-base Search Engine</span>
    </div>''',
    unsafe_allow_html=True
)

st.write("")

# --- Main UI ---
with st.container():
    st.markdown(
        "<h3 style='text-align:center; color:#3B82F6;'>Upload a PDF, ask a question, and get a smart answer!</h3>",
        unsafe_allow_html=True
    )
    st.write("")
    with st.sidebar:
        st.header("Model options")
        models_dir = st.text_input("Local models directory (optional)", value="models")

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    question = st.text_input("Enter your question", key="question_input")

    if uploaded_file is not None:
        if 'engine' not in st.session_state:
            try:
                st.session_state['engine'] = RAGEngine(local_models_dir=models_dir)
            except Exception as e:
                st.error(f"Error initializing models: {e}")

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
                    st.markdown("<h3 style='color:#3B82F6;'>Answer</h3>", unsafe_allow_html=True)
                    st.write(summary)

                    st.markdown("<h4 style='color:#3B82F6;'>Top source snippets</h4>", unsafe_allow_html=True)
                    for i, (idx, score, chunk) in enumerate(top_chunks, start=1):
                        st.markdown(f"**{i}. (score: {score:.4f})**")
                        st.write(chunk[:1000])
                except Exception as e:
                    st.error(f"Error generating answer: {e}")

    # --- New feature buttons ---
    col1, col2, col3 = st.columns(3)
    with col1:
        summarize_btn = st.button("Summarize Document", use_container_width=True)
    with col2:
        topics_btn = st.button("Important Topics", use_container_width=True)
    with col3:
        quiz_btn = st.button("Quiz Me", use_container_width=True)

    if uploaded_file is not None and 'engine' in st.session_state:
        engine = st.session_state['engine']
        # Summarize Document
        if summarize_btn:
            with st.spinner("Summarizing document..."):
                try:
                    context = "\n\n".join(engine.chunks)
                    if len(context) > 4000:
                        context = context[:4000]
                    result = engine.summarizer(f"Summarize this document:\n{context}", max_length=180, min_length=50, do_sample=False)
                    summary = result[0]["summary_text"].strip()
                    st.markdown("<h3 style='color:#3B82F6;'>Document Summary</h3>", unsafe_allow_html=True)
                    st.write(summary)
                except Exception as e:
                    st.error(f"Error summarizing: {e}")
        # Important Topics
        if topics_btn:
            with st.spinner("Extracting important topics..."):
                try:
                    topics = engine.extract_topics(top_n=5)
                    st.markdown("<h3 style='color:#3B82F6;'>Important Topics</h3>", unsafe_allow_html=True)
                    if isinstance(topics, list):
                        st.markdown("\n".join([f"- {t}" for t in topics]))
                    else:
                        st.markdown(topics)
                except Exception as e:
                    st.error(f"Error extracting topics: {e}")
        # Quiz Me
        if quiz_btn:
            with st.spinner("Generating quiz questions..."):
                try:
                    quiz_pairs = engine.generate_quiz(num_questions=3)
                    st.markdown("<h3 style='color:#3B82F6;'>Quiz Me</h3>", unsafe_allow_html=True)
                    if isinstance(quiz_pairs, list) and all(isinstance(q, tuple) for q in quiz_pairs):
                        for i, (q, a) in enumerate(quiz_pairs, 1):
                            st.markdown(f"**Q{i}:** {q}")
                            st.markdown(f"<span style='color:#64748B'><b>Answer:</b> {a}</span>", unsafe_allow_html=True)
                            st.write("")
                    else:
                        st.markdown(quiz_pairs[0])
                except Exception as e:
                    st.error(f"Error generating quiz: {e}")

st.markdown("---")
st.caption("All processing is local. Models are loaded from the `models/` directory if present; otherwise the library may try to download them the first time.")

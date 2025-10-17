# Offline Knowledge-base Search Engine (RAG)

This repository demonstrates a small, fully offline Retrieval-Augmented Generation (RAG) application using only local, free components.

Goal
------
Let a user upload a single PDF, extract text, chunk it, embed the chunks with a local SentenceTransformer, run a semantic search by cosine similarity, and produce a concise answer using a local HuggingFace summarization model — all on CPU and without any paid APIs.

Key features
------------
- Upload a single PDF via a Streamlit UI
- Extract text using PyPDF2
- Split text into ~500-word chunks
- Create embeddings with `sentence-transformers` (`all-MiniLM-L6-v2`)
- Semantic search via cosine similarity (scikit-learn)
- Summarize top 3 retrieved chunks using a local `facebook/bart-large-cnn` model
- Offline-first: after initial model downloads, the app runs without internet or API keys

Files
------
- `ui_app.py` — Streamlit UI (entrypoint)
- `rag_engine.py` — core logic: PDF extraction, chunking, embeddings, search, summarization
- `requirements.txt` — Python dependencies

Quick notes about offline use
-----------------------------
- The first time you run the app (or run the helper download commands below), the models will be downloaded from Hugging Face. After those models are cached or explicitly saved to a local `models/` folder, the app can be run 100% offline.
- If you need strictly no internet at all during setup, follow the manual download commands on a machine with internet and copy the `models/` folder to the target offline machine.

Setup (Windows PowerShell)
--------------------------
Open PowerShell and run:

```powershell
# create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# install dependencies
pip install -r requirements.txt
```

(If you prefer cmd.exe: use `.venv\Scripts\activate.bat`)

Optional: pre-download models so the app can run offline
-------------------------------------------------------
Run these commands once on a machine with internet access. They will save the models under a local `models/` directory. Copy that `models/` folder to your offline machine (place it inside the project root).

```powershell
# Pre-download the sentence-transformers model
python - << 'PY'
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('all-MiniLM-L6-v2')
m.save('models/all-MiniLM-L6-v2')
print('Saved sentence-transformers model')
PY

# Pre-download the summarization model (tokenizer + model)
python - << 'PY'
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model_name = 'facebook/bart-large-cnn'
AutoTokenizer.from_pretrained(model_name).save_pretrained('models/facebook-bart-large-cnn')
AutoModelForSeq2SeqLM.from_pretrained(model_name).save_pretrained('models/facebook-bart-large-cnn')
print('Saved summarization model')
PY
```

How the app uses these local models
----------------------------------
In `rag_engine.py` the SentenceTransformer can be loaded with either the model name (when online) or the local path `'models/all-MiniLM-L6-v2'`. Similarly, the summarizer pipeline can be created with `model='models/facebook-bart-large-cnn'` so it uses the local files.

Run the app
-----------
Once dependencies are installed and models are present (or have been downloaded once), start the Streamlit app with:

```powershell
streamlit run ui_app.py
```

Usage
-----
- Open the Streamlit page (Streamlit prints the local URL to the terminal).
- Upload a single PDF.
- Enter a question in the text input.
- Click "Generate Answer".
- The app will show a concise summarized answer and the top 3 source snippets.

Tips and troubleshooting
------------------------
- If `torch` installation is problematic on Windows, install the CPU-only wheel per the PyTorch instructions (choose your Python version and run the command from https://pytorch.org/get-started/locally/). Example for many setups:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

- If you want to reduce memory usage, you can switch to a smaller summarization model (for example `sshleifer/distilbart-cnn-12-6`) — but `facebook/bart-large-cnn` gives better quality.

- If a PDF has scanned pages (images), PyPDF2 won't extract readable text. Convert scanned pages with OCR (outside the scope of this demo).

Security
--------
- This project is offline-first and does not call any external APIs by default. Do not commit large downloaded model files to git. Instead, add `models/` to `.gitignore`.

License & attribution
----------------------
This project uses open-source models and libraries. Respect the respective licenses for `sentence-transformers`, `transformers`, and model-specific licenses.

Next steps
----------
- Implement `ui_app.py` and `rag_engine.py` (I can scaffold these next).
- Add a `.gitignore` and a small sample PDF for demos.

If you'd like, I'll now scaffold `rag_engine.py` and `ui_app.py` so you can run the app immediately. Let me know and I'll create the files and run a quick smoke-check (no internet required if you pre-downloaded models).
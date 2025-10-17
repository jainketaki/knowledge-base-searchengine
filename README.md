RAG Search Engine 🔍🤖
<img width="1918" height="852" alt="image" src="https://github.com/user-attachments/assets/925d2579-4c73-46bd-ab98-fcb643856340" />


A production-ready Retrieval-Augmented Generation (RAG) system that transforms how you search through documents. Upload your files, ask questions in natural language, and get AI-powered answers with source citations - all in under 3 seconds.

What is RAG? Retrieval-Augmented Generation means the AI doesn't just guess - it first retrieves relevant information from your actual documents, then generates accurate answers based on what it found. Think of it like giving ChatGPT the ability to read your specific files.


🎥 Demo
Watch the video here: https://drive.google.com/file/d/1-fe4OAJ7rpQjEcBOJDqMO-ir8H_IU-zv/view?usp=sharing 


✨ Why This Matters
Traditional Search Problems:

🔴 Keyword matching misses relevant results
🔴 No direct answers, just document lists
🔴 Can't synthesize information from multiple sources
🔴 No way to verify where information came from

RAG Search Engine Solutions:

✅ Understands meaning, not just keywords (semantic search)
✅ Provides direct answers in natural language
✅ Combines information from multiple documents
✅ Cites sources for every claim
✅ Response time under 3 seconds


🚀 Core Features
📄 Document Processing & Management

Multi-Format Support - Upload PDFs, TXT, and Markdown files
Intelligent Chunking - Automatically splits documents into 512-token chunks with overlap
Metadata Tracking - Maintains source file, chunk index, and timestamps
Batch Processing - Handle multiple documents efficiently
Document Management - List, view, and delete indexed documents

🔍 Smart Search & Retrieval

Semantic Vector Search - Understands meaning using ChromaDB embeddings
Fast Retrieval - Sub-200ms average search time
Relevance Scoring - Each result scored 0-1 for accuracy
Query Reranking - Improves results with keyword overlap analysis
Configurable Results - Return top 1-20 most relevant chunks
Metadata Filtering - Filter by document source or date

🤖 AI-Powered Answer Synthesis

GPT-4 Integration - High-quality answer generation
Source Attribution - Every answer cites specific documents and pages
Multi-Document Synthesis - Combines information from multiple sources
Context-Aware Prompting - Structured prompts for accurate responses
Handles Missing Info - Clearly states when documents don't contain the answer
Confidence Scoring - Assesses answer quality based on relevance

🛠️ Production-Ready Infrastructure

FastAPI Backend - Modern, async REST API
Docker Containerization - One-command deployment
Health Monitoring - Built-in health check endpoints
Comprehensive Testing - 80%+ code coverage with pytest
Error Handling - Robust exception management throughout
Logging System - Structured logging for debugging

⚡ Performance & Optimization

Caching - Reduces API costs and improves speed
Sub-3-Second Responses - Fast retrieval + synthesis
Scales to 1000+ Documents - Handles large knowledge bases
Efficient Chunking - Optimized token usage
Connection Pooling - Database optimization

🌐 Deployment & DevOps

Cloud-Ready - Deploy to AWS, GCP, Azure, or Heroku
Docker Compose - Multi-container orchestration
Environment Configuration - 12-factor app design
CI/CD Ready - Easy integration with pipelines
Multiple Deployment Guides - Step-by-step for each platform


📊 Key Metrics
MetricPerformanceRetrieval Time< 200ms averageTotal Response Time< 3 secondsDocument Capacity1000+ documentsRelevance Score85%+ averageTest Coverage80%+Supported FormatsPDF, TXT, MD

🏗️ Architecture
┌─────────────────────────────────────────────────────────────┐
│                     USER UPLOADS DOCUMENT                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │   TEXT EXTRACTION      │
         │   (PyPDF2/pdfplumber)  │
         └────────┬───────────────┘
                  │
                  ▼
         ┌────────────────────────┐
         │   INTELLIGENT CHUNKING │
         │   (512 tokens + overlap)│
         └────────┬───────────────┘
                  │
                  ▼
         ┌────────────────────────┐
         │  EMBEDDING GENERATION  │
         │  (OpenAI text-embed-3) │
         └────────┬───────────────┘
                  │
                  ▼
         ┌────────────────────────┐
         │   CHROMADB STORAGE     │
         │   (Vector Database)    │
         └────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                       USER ASKS QUESTION                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │   QUERY EMBEDDING      │
         │   (Convert to vector)  │
         └────────┬───────────────┘
                  │
                  ▼
         ┌────────────────────────┐
         │   SEMANTIC SEARCH      │
         │   (Cosine similarity)  │
         │   Retrieval: ~150ms    │
         └────────┬───────────────┘
                  │
                  ▼
         ┌────────────────────────┐
         │  RETRIEVE TOP-K CHUNKS │
         │  (Relevance ranked)    │
         └────────┬───────────────┘
                  │
                  ▼
         ┌────────────────────────┐
         │   CONTEXT ASSEMBLY     │
         │   (Combine chunks)     │
         └────────┬───────────────┘
                  │
                  ▼
         ┌────────────────────────┐
         │   GPT-4 SYNTHESIS      │
         │   Generation: ~1.5s    │
         └────────┬───────────────┘
                  │
                  ▼
         ┌────────────────────────┐
         │   RETURN ANSWER        │
         │   WITH SOURCES         │
         └────────────────────────┘
Made with love for Unthinkable.
Author:
Ketaki Jain

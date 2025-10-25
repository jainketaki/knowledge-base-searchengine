"""
Core RAG engine: PDF extraction, chunking, embedding, search, summarization.
Designed for offline use with local models saved under `models/` (optional).
Fixed quiz generation logic to work properly.
"""
from typing import List, Tuple
import os
import re
import random

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
            if txt:
                # Basic cleaning of PDF extraction artifacts
                # Fix common issues where spaces are missing between words
                txt = re.sub(r'([a-z])([A-Z])', r'\1 \2', txt)  # camelCase to proper spacing
                txt = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', txt)  # Punctuation spacing
                txt = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', txt)  # Letter-number spacing
                txt = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', txt)  # Number-letter spacing
                texts.append(txt)
        except Exception:
            continue
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
        # First, handle common PDF extraction issues
        # Fix missing spaces between words that got concatenated
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between lowercase and uppercase
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)  # Add space between letters and numbers
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)  # Add space between numbers and letters
        text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)  # Ensure space after punctuation
        text = re.sub(r'([a-z])([.!?])', r'\1 \2', text)  # Space before punctuation if missing
        
        # Handle common PDF artifacts
        text = re.sub(r'([a-z])(\()', r'\1 \2', text)  # Space before opening parenthesis
        text = re.sub(r'(\))([a-zA-Z])', r'\1 \2', text)  # Space after closing parenthesis
        text = re.sub(r'([a-z])(,)', r'\1\2', text)  # Keep comma attached to word
        text = re.sub(r'(,)([a-zA-Z])', r'\1 \2', text)  # Space after comma
        
        # Replace multiple spaces/newlines with a single space
        text = re.sub(r'[ \t\r\f\v]+', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        
        # Fix common word concatenations
        text = re.sub(r'([a-z])(the|and|or|in|on|at|to|for|of|with|by)', r'\1 \2', text)
        text = re.sub(r'(the|and|or|in|on|at|to|for|of|with|by)([A-Z][a-z])', r'\1 \2', text)
        
        return text.strip()

    def _final_text_cleanup(self, text: str) -> str:
        """Final cleanup pass to ensure proper spacing and readability."""
        if not text:
            return text
            
        # Fix common spacing issues
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
        text = re.sub(r'\s*([.!?])\s*', r'\1 ', text)  # Proper spacing around sentence endings
        text = re.sub(r'\s*,\s*', r', ', text)  # Proper spacing around commas
        text = re.sub(r'\s*;\s*', r'; ', text)  # Proper spacing around semicolons
        text = re.sub(r'\s*:\s*', r': ', text)  # Proper spacing around colons
        
        # Fix parentheses spacing
        text = re.sub(r'\s*\(\s*', r' (', text)
        text = re.sub(r'\s*\)\s*', r') ', text)
        
        # Fix quotation marks
        text = re.sub(r'\s*"\s*', r' "', text)
        text = re.sub(r'"\s*', r'" ', text)
        
        # Remove extra spaces at the beginning and end
        text = text.strip()
        
        # Ensure sentences start with capital letters
        sentences = re.split(r'([.!?]\s+)', text)
        cleaned_sentences = []
        for i, sentence in enumerate(sentences):
            if i % 2 == 0 and sentence.strip():  # Actual sentence content
                sentence = sentence.strip()
                if sentence and sentence[0].islower():
                    sentence = sentence[0].upper() + sentence[1:]
                cleaned_sentences.append(sentence)
            else:
                cleaned_sentences.append(sentence)
        
        return ''.join(cleaned_sentences).strip()

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

    def query(self, question: str, top_k: int = 5) -> Tuple[str, List[Tuple[int, float, str]]]:
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
        
        # Create a comprehensive summary using multiple approaches
        summary = self._create_comprehensive_summary(context, question)
        
        return summary, top_chunks

    def _create_comprehensive_summary(self, context: str, question: str = "") -> str:
        """Create a comprehensive summary using multiple techniques."""
        # Clean the context first to fix spacing issues
        context = self._clean_text(context)
        
        # Method 1: Try using the summarizer with better parameters
        summary_parts = []
        
        # Split context into manageable chunks for the summarizer
        context_chunks = []
        words = context.split()
        chunk_size = 800  # Larger chunks for better context
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            context_chunks.append(chunk)
        
        # Summarize each chunk
        for i, chunk in enumerate(context_chunks[:3]):  # Limit to 3 chunks to avoid too much processing
            try:
                if question:
                    prompt = f"Summarize the following text in relation to this question: {question}\n\nText: {chunk}"
                else:
                    prompt = f"Provide a detailed summary of the following text:\n\n{chunk}"
                
                result = self.summarizer(prompt,
                                       max_length=250,  # Increased max length
                                       min_length=80,   # Increased min length
                                       do_sample=False,
                                       num_beams=2,     # Better quality with more beams
                                       temperature=1.0)
                chunk_summary = result[0]["summary_text"].strip()
                chunk_summary = self._clean_text(chunk_summary)
                
                # Filter out poor quality summaries
                if (chunk_summary and 
                    len(chunk_summary.split()) >= 15 and 
                    not chunk_summary.lower().startswith("question:") and
                    not chunk_summary.lower().startswith("summarize") and
                    not chunk_summary.lower().startswith("provide")):
                    summary_parts.append(chunk_summary)
                    
            except Exception as e:
                print(f"Summarization error for chunk {i}: {str(e)}")
                continue
        
        # Method 2: If summarizer fails or produces poor results, use extractive approach
        if not summary_parts or sum(len(part.split()) for part in summary_parts) < 50:
            summary_parts = self._extractive_summary(context, target_sentences=8)
        
        # Combine and clean up the summary
        if summary_parts:
            final_summary = " ".join(summary_parts)
            final_summary = self._clean_text(final_summary)
            
            # Ensure minimum length
            if len(final_summary.split()) < 30:
                # Add more context if summary is too short
                additional_context = self._extractive_summary(context, target_sentences=5)
                final_summary += " " + " ".join(additional_context)
                final_summary = self._clean_text(final_summary)
        else:
            # Final fallback: use the first portion of context
            context_clean = self._clean_text(context)
            final_summary = context_clean[:1000] + ("..." if len(context_clean) > 1000 else "")
        
        # Final cleaning pass to ensure proper spacing
        final_summary = self._final_text_cleanup(final_summary)
        return final_summary

    def _extractive_summary(self, text: str, target_sentences: int = 6) -> List[str]:
        """Create an extractive summary by selecting the most important sentences."""
        # Clean the text first
        text = self._clean_text(text)
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter and score sentences
        scored_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence.split()) >= 8:  # Only meaningful sentences
                # Simple scoring based on length and content
                score = len(sentence.split())
                
                # Boost score for sentences with important indicators
                important_words = ['important', 'key', 'main', 'primary', 'significant', 
                                 'essential', 'critical', 'fundamental', 'major', 'central']
                for word in important_words:
                    if word in sentence.lower():
                        score += 10
                
                # Boost score for sentences with numbers/statistics
                if re.search(r'\d+', sentence):
                    score += 5
                
                # Clean the sentence before adding
                clean_sentence = self._final_text_cleanup(sentence)
                scored_sentences.append((score, clean_sentence))
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [sent for score, sent in scored_sentences[:target_sentences]]
        
        return top_sentences

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

    def _extract_key_sentences(self, text: str, min_words: int = 8) -> List[str]:
        """Extract meaningful sentences that could be turned into questions."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        key_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            # Filter out short sentences, headings, and low-content sentences
            if (len(sentence.split()) >= min_words and 
                len(sentence.split()) <= 50 and  # Not too long either
                not sentence.isupper() and  # Skip ALL CAPS headings
                not re.match(r'^\d+\.?\s*$', sentence) and  # Skip numbered items
                not sentence.startswith('Figure') and
                not sentence.startswith('Table') and
                not sentence.startswith('Page') and
                not sentence.startswith('Chapter') and
                '.' in sentence and  # Ensure it's a complete sentence
                not sentence.startswith('www.') and  # Skip URLs
                not sentence.startswith('http')):  # Skip URLs
                key_sentences.append(sentence)
        
        return key_sentences

    def _create_question_from_sentence(self, sentence: str) -> Tuple[str, str]:
        """Convert a sentence into a question-answer pair with improved patterns."""
        sentence = sentence.strip()
        
        # Pattern 1: "X is Y" -> "What is X?"
        match = re.search(r'([A-Z][^.]*?)\s+(?:is|are|was|were)\s+([^.]+)', sentence)
        if match:
            subject = match.group(1).strip()
            predicate = match.group(2).strip()
            if len(subject.split()) <= 6 and len(subject.split()) >= 1:
                question = f"What is {subject}?"
                answer = f"{subject} is {predicate}."
                return question, answer
        
        # Pattern 2: "X involves/includes/contains Y" -> "What does X involve/include?"
        match = re.search(r'([A-Z][^.]*?)\s+(involves?|includes?|contains?|comprises?|consists?\s+of)\s+([^.]+)', sentence)
        if match:
            subject = match.group(1).strip()
            verb = match.group(2).strip()
            object_part = match.group(3).strip()
            if len(subject.split()) <= 6:
                question = f"What does {subject} {verb.lower()}?"
                answer = f"{subject} {verb.lower()} {object_part}."
                return question, answer
        
        # Pattern 3: "The purpose/goal/objective of X is Y" -> "What is the purpose of X?"
        match = re.search(r'[Tt]he\s+(purpose|goal|objective|aim)\s+of\s+([^.]*?)\s+is\s+([^.]+)', sentence)
        if match:
            purpose_word = match.group(1)
            subject = match.group(2).strip()
            purpose = match.group(3).strip()
            question = f"What is the {purpose_word} of {subject}?"
            answer = f"The {purpose_word} of {subject} is {purpose}."
            return question, answer
        
        # Pattern 4: "X can/should/must/will Y" -> "What can/should/must/will X do?"
        match = re.search(r'([A-Z][^.]*?)\s+(can|should|must|will|may|might)\s+([^.]+)', sentence)
        if match:
            subject = match.group(1).strip()
            modal = match.group(2).strip()
            action = match.group(3).strip()
            if len(subject.split()) <= 6:
                question = f"What {modal} {subject} do?"
                answer = f"{subject} {modal} {action}."
                return question, answer
        
        # Pattern 5: Numbers and statistics
        match = re.search(r'([^.]*?)\s+(?:is|are|was|were)\s+(\d+[%\w\s]*)', sentence)
        if match:
            context = match.group(1).strip()
            number = match.group(2).strip()
            if len(context.split()) <= 8:
                question = f"What is {context}?"
                answer = f"{context} is {number}."
                return question, answer
        
        # Pattern 6: "X happens/occurs when Y" -> "When does X happen?"
        match = re.search(r'([^.]*?)\s+(happens?|occurs?|takes?\s+place)\s+when\s+([^.]+)', sentence)
        if match:
            event = match.group(1).strip()
            condition = match.group(3).strip()
            question = f"When does {event} happen?"
            answer = f"{event} happens when {condition}."
            return question, answer
        
        # Pattern 7: "There are X types/kinds/categories of Y" -> "How many types of Y are there?"
        match = re.search(r'[Tt]here\s+are\s+(\w+)\s+(types?|kinds?|categories?)\s+of\s+([^.]+)', sentence)
        if match:
            number = match.group(1)
            category_word = match.group(2)
            subject = match.group(3).strip()
            question = f"How many {category_word} of {subject} are there?"
            answer = f"There are {number} {category_word} of {subject}."
            return question, answer
        
        # Pattern 8: "X allows/enables/helps Y" -> "What does X allow/enable/help?"
        match = re.search(r'([A-Z][^.]*?)\s+(allows?|enables?|helps?)\s+([^.]+)', sentence)
        if match:
            subject = match.group(1).strip()
            verb = match.group(2).strip()
            object_part = match.group(3).strip()
            if len(subject.split()) <= 6:
                question = f"What does {subject} {verb.lower()}?"
                answer = f"{subject} {verb.lower()} {object_part}."
                return question, answer
        
        # Pattern 9: "X results in/leads to Y" -> "What does X result in?"
        match = re.search(r'([A-Z][^.]*?)\s+(results?\s+in|leads?\s+to|causes?)\s+([^.]+)', sentence)
        if match:
            subject = match.group(1).strip()
            verb = match.group(2).strip()
            result = match.group(3).strip()
            if len(subject.split()) <= 6:
                question = f"What does {subject} {verb.lower()}?"
                answer = f"{subject} {verb.lower()} {result}."
                return question, answer
        
        # Pattern 10: Generic fallback for definition-like sentences
        if len(sentence.split()) >= 10 and len(sentence.split()) <= 25:
            # Try to extract the main subject (first few words before first verb)
            words = sentence.split()
            for i, word in enumerate(words[:7]):
                if word.lower() in ['is', 'are', 'was', 'were', 'can', 'will', 'should', 'must', 'may', 'might', 'has', 'have']:
                    subject = ' '.join(words[:i])
                    if len(subject.split()) >= 2 and len(subject.split()) <= 6:
                        question = f"What can you tell me about {subject}?"
                        answer = sentence
                        return question, answer
                    break
        
        return None, None

    def generate_quiz(self, num_questions: int = 3) -> list:
        """Generate quiz questions from the document content using rule-based approach."""
        if not self.chunks or len(self.chunks) == 0:
            return [("No Content", "Please upload a PDF document first to generate quiz questions.")]
        
        # Combine all chunks and clean the text
        full_text = " ".join(self.chunks)
        full_text = self._clean_text(full_text)
        
        # Extract key sentences with lower threshold for more variety
        key_sentences = self._extract_key_sentences(full_text, min_words=6)
        
        if not key_sentences:
            return [("No Suitable Content", "The document doesn't contain enough structured content to generate quiz questions.")]
        
        # Shuffle sentences to get variety
        random.shuffle(key_sentences)
        
        # Generate questions from sentences
        qa_pairs = []
        used_sentences = set()
        used_questions = set()
        
        # Try to generate questions from different parts of the document
        max_attempts = min(len(key_sentences), num_questions * 10)  # Try more sentences
        
        for i, sentence in enumerate(key_sentences[:max_attempts]):
            if len(qa_pairs) >= num_questions:
                break
                
            if sentence in used_sentences:
                continue
                
            question, answer = self._create_question_from_sentence(sentence)
            
            if question and answer and question not in used_questions:
                # Clean both question and answer for proper spacing
                question = self._final_text_cleanup(question)
                answer = self._final_text_cleanup(answer)
                
                # Ensure question and answer are of good quality
                if (len(question.split()) >= 4 and 
                    len(answer.split()) >= 5 and
                    len(answer.split()) <= 50):
                    qa_pairs.append((question, answer))
                    used_sentences.add(sentence)
                    used_questions.add(question)
        
        # If we don't have enough questions, create topic-based questions
        if len(qa_pairs) < num_questions:
            remaining_needed = num_questions - len(qa_pairs)
            topic_questions = self._generate_topic_questions(remaining_needed)
            qa_pairs.extend(topic_questions)
        
        # If still not enough, create chunk-based questions
        if len(qa_pairs) < num_questions:
            remaining_needed = num_questions - len(qa_pairs)
            chunk_questions = self._generate_chunk_questions(remaining_needed)
            qa_pairs.extend(chunk_questions)
        
        return qa_pairs[:num_questions]

    def _generate_topic_questions(self, num_needed: int) -> List[Tuple[str, str]]:
        """Generate questions based on document topics."""
        qa_pairs = []
        topics = self.extract_topics(top_n=num_needed * 2)
        
        for topic in topics:
            if len(qa_pairs) >= num_needed:
                break
                
            if isinstance(topic, str) and len(topic.strip()) > 3:
                topic_clean = topic.strip()
                
                # Find the best sentence containing this topic
                best_sentence = None
                best_score = 0
                
                for chunk in self.chunks:
                    clean_chunk = self._clean_text(chunk)
                    sentences = re.split(r'(?<=[.!?])\s+', clean_chunk)
                    for sentence in sentences:
                        if (topic_clean.lower() in sentence.lower() and 
                            len(sentence.split()) >= 8 and 
                            len(sentence.split()) <= 40):
                            # Score based on sentence quality
                            score = len(sentence.split()) + sentence.lower().count(topic_clean.lower()) * 5
                            if score > best_score:
                                best_score = score
                                best_sentence = sentence
                
                if best_sentence:
                    question = f"What does the document say about {topic_clean}?"
                    answer = self._final_text_cleanup(best_sentence.strip())
                    qa_pairs.append((question, answer))
                    
        return qa_pairs

    def _generate_chunk_questions(self, num_needed: int) -> List[Tuple[str, str]]:
        """Generate questions from important document chunks."""
        qa_pairs = []
        
        # Sort chunks by length (assuming longer chunks have more content)
        chunk_scores = [(i, len(chunk), chunk) for i, chunk in enumerate(self.chunks)]
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        for i, (chunk_idx, length, chunk) in enumerate(chunk_scores[:num_needed]):
            if len(qa_pairs) >= num_needed:
                break
                
            # Clean the chunk first
            clean_chunk = self._clean_text(chunk)
            
            # Create a general question about this section
            question = f"What is discussed in section {i+1} of the document?"
            
            # Get the first few sentences as answer
            sentences = re.split(r'(?<=[.!?])\s+', clean_chunk)
            good_sentences = [s for s in sentences[:3] if len(s.split()) >= 5]
            
            if good_sentences:
                answer = ". ".join(good_sentences[:2])
                if len(answer) > 300:
                    answer = answer[:297] + "..."
                answer = self._final_text_cleanup(answer)
                qa_pairs.append((question, answer))
        
        return qa_pairs


if __name__ == "__main__":
    print("rag_engine.py: basic smoke test")
    # This basic test only checks imports and class creation (no models downloaded here)
    try:
        engine = RAGEngine()
        print("RAGEngine created (models may attempt to download if not present)." )
    except Exception as e:
        print("Error creating RAGEngine:", e)
# 🏛️ UPSC Prelims AI Mentor (RAG-Powered MCQ Engine)

[Streamlit App] - https://upsc-rag-mentor-m5dkqjqy3es6xxa8ojrdzh.streamlit.app 

An end-to-end Retrieval-Augmented Generation (RAG) platform tailored for the **UPSC Civil Services Preliminary Examination (General Studies Paper I)**. The engine indexes high-yield syllabus materials, committee reports, and current affairs compilations into a serverless vector database, generating multi-statement Prelims questions grounded strictly in the source text.

---

## 🎯 Project Objective

Standard commercial LLMs frequently suffer from hallucinations or produce simplistic, trivia-based multiple choice questions unsuitable for UPSC Civil Services preparation. 

This project solves this by:
* **Grounded Question Synthesis:** Restricting question generation exclusively to retrieved vector chunks from verified reference PDFs (eliminating hallucinated facts, articles, and provisions).
* **Authentic UPSC Prelims Pattern:** Modeling the structural rigor of UPSC GS-1 papers—featuring 2–3 complex analytical statements, directional phrases (*"Which of the statements given above is/are correct?"*), and standard options (*"1 only"*, *"Both 1 and 2"*, etc.).
* **Active Feedback & Marking Scheme:** Evaluating responses in real-time under actual UPSC marking criteria (+2.00 marks for correct answers, -0.66 marks for wrong answers) alongside comprehensive statement-by-statement rationales and source chunk citations.
* **Corpus & Chunk Transparency:** Providing built-in chunk inspection and vector corpus management directly within the UI.

---

## 🛠️ Tech Stack & Infrastructure

* **Frontend / Application Framework:** [Streamlit](https://streamlit.io/)
* **Vector Store:** [Pinecone Serverless](https://www.pinecone.io/) (Dimension: 768, Metric: Cosine)
* **Embeddings Model:** Google `gemini-embedding-001` (768-dimensional output)
* **Large Language Models:** Google Gemini via `google-genai` SDK
  * `gemini-2.5-flash` (Primary production generation)
  * `gemini-2.0-flash` / `gemini-1.5-flash` / `gemini-3.8-flash` (Automated fallback cascade)
* **Document Extraction:** `pypdf` (In-memory streaming, chunk extraction, and ingestion)
* **Local Caching Layer:** SQLite (`query_cache.db`) for semantic query vector caching (reducing API call overhead and preserving quotas)
* **Hosting & CI/CD:** Streamlit Community Cloud linked directly to GitHub (`BALAJIKOWSHIK/Upsc-Rag-mentor`)

---

## 📐 Architecture & Workflow

```text
               +----------------------------------+
               |  UPSC Reference PDFs / Monthly   |
               +----------------------------------+
                                │
                                ▼ [Chunking: 800 chars / 100 overlap]
               +----------------------------------+
               |        Syllabus Chunks           |
               +----------------------------------+
                                │
                                ▼ [gemini-embedding-001 (768d)]
               +----------------------------------+
               |       Pinecone Serverless        |
               +----------------------------------+
                                ▲
                                │  Cosine Similarity (Top-K)
                                │
[User Topic Query] ──► [SQLite Vector Cache] ──(Miss)──► [Query Embedding]
                                │
                                ▼ [Retrieved Chunks]
               +----------------------------------+
               |    Context-Injected System Prompt|
               +----------------------------------+
                                │
                                ▼ [Gemini Fallback Cascade + JSON Mode]
               +----------------------------------+
               | Structured UPSC Prelims Test UI  |
               +----------------------------------+

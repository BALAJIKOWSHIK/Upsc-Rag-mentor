import streamlit as st
import os
import json
import sqlite3
import tomllib
from google import genai
from pinecone import Pinecone

st.set_page_config(
    page_title="UPSC Prelims AI Mentor",
    page_icon="🏛️",
    layout="wide"
)

# ---------------------------------------------------------
# Credential Loading (Prioritizes .streamlit/secrets.toml)
# ---------------------------------------------------------
def get_secret(key):
    try:
        if os.path.exists(".streamlit/secrets.toml"):
            with open(".streamlit/secrets.toml", "rb") as f:
                data = tomllib.load(f)
                val = data.get(key)
                if val:
                    return val
    except Exception:
        pass
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key)

google_key = get_secret("GEMINI_API_KEY")
pinecone_key = get_secret("PINECONE_API_KEY")

if not google_key or not pinecone_key:
    st.error("Missing GEMINI_API_KEY or PINECONE_API_KEY in `.streamlit/secrets.toml`.")
    st.stop()

# ---------------------------------------------------------
# Native SDK Clients Setup
# ---------------------------------------------------------
@st.cache_resource
def get_clients():
    ai_client = genai.Client(api_key=google_key)
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index("upsc-mentor")
    return ai_client, index

ai_client, index = get_clients()

# ---------------------------------------------------------
# Local SQLite Vector Cache (Prevents Quota Burn)
# ---------------------------------------------------------
def get_cached_vector(query_text):
    conn = sqlite3.connect("query_cache.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vector_cache (
            query TEXT PRIMARY KEY,
            vector TEXT
        )
    """)
    cursor.execute("SELECT vector FROM vector_cache WHERE query = ?", (query_text.strip().lower(),))
    row = cursor.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return None

def save_vector_to_cache(query_text, vector):
    conn = sqlite3.connect("query_cache.db")
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO vector_cache (query, vector) VALUES (?, ?)", 
                   (query_text.strip().lower(), json.dumps(vector)))
    conn.commit()
    conn.close()

# ---------------------------------------------------------
# Pinecone Context Retrieval
# ---------------------------------------------------------
def retrieve_chunks(topic_query, top_k=5):
    query_vector = get_cached_vector(topic_query)
    
    if not query_vector:
        try:
            embed_resp = ai_client.models.embed_content(
                model="models/gemini-embedding-001",
                contents=topic_query,
                config={"output_dimensionality": 768}
            )
            query_vector = embed_resp.embeddings[0].values
            save_vector_to_cache(topic_query, query_vector)
        except Exception as e:
            st.warning(f"Embedding API limit hit. Generating test using base syllabus knowledge.")
            return []

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    
    chunks = [match["metadata"]["text"] for match in results.get("matches", []) if "text" in match.get("metadata", {})]
    return chunks

# ---------------------------------------------------------
# Resilient LLM Generator with Fallback Cascade
# ---------------------------------------------------------
def generate_mcqs_from_context(topic, context_chunks, num_questions=3):
    candidate_models = [
        "gemini-3.5-flash-lite",
        "gemini-flash-latest",
        "gemini-3.8-flash"
    ]
    
    context_text = "\n\n---\n\n".join(context_chunks) if context_chunks else "Standard UPSC Civil Services GS Syllabus."
    
    system_prompt = f"""You are an expert UPSC Civil Services Prelims examination creator.
Your task is to generate {num_questions} authentic, high-caliber UPSC Prelims-style Multiple Choice Questions based strictly on the provided context chunks.

Requirements for each question:
1. Strict UPSC Prelims format: Usually 2 or 3 statements, followed by standard combinations (e.g., '1 only', '2 only', 'Both 1 and 2', 'Neither 1 nor 2', or 'Which of the statements given above is/are correct?').
2. Highly educational: The question must test deep conceptual application, landmark cases, constitutional articles, or institutional mechanisms found in the context.
3. Detailed Explanations: Explain why each individual statement is correct or incorrect based directly on the provided context.

You MUST respond strictly with a valid JSON array of objects without Markdown formatting or backticks around the json.
Schema for each JSON object:
{{
  "id": 1,
  "question_stem": "Consider the following statements regarding ...",
  "statements": [
    "1. First statement text.",
    "2. Second statement text."
  ],
  "directive": "Which of the statements given above is/are correct?",
  "options": [
    "A) 1 only",
    "B) 2 only",
    "C) Both 1 and 2",
    "D) Neither 1 nor 2"
  ],
  "correct_option": "A",
  "explanation": "Detailed breakdown explaining why Statement 1 is correct and Statement 2 is incorrect based on the syllabus chunk.",
  "grounded_chunk_excerpt": "Relevant sentence or principle from the source material."
}}

Context:
{context_text}

Topic:
{topic}
"""

    last_error = None
    for model_name in candidate_models:
        try:
            resp = ai_client.models.generate_content(
                model=model_name,
                contents=system_prompt,
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0.2
                }
            )
            data = json.loads(resp.text)
            return data
        except Exception as e:
            last_error = e
            continue

    st.error(f"Failed to generate questions across available models: {last_error}")
    return []

# ---------------------------------------------------------
# UI Layout & Session State
# ---------------------------------------------------------
st.sidebar.title("🏛️ UPSC Prelims Studio")
st.sidebar.caption("Grounding questions directly on your Pinecone vector database chunks.")

study_mode = st.sidebar.radio(
    "Select Practice Mode:",
    ["Topic-wise Drill", "Browse Document Chunks"]
)

if "active_quiz" not in st.session_state:
    st.session_state.active_quiz = None
if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}
if "submitted" not in st.session_state:
    st.session_state.submitted = False

# ---------------------------------------------------------
# View 1: Topic-wise Prelims MCQ Drill
# ---------------------------------------------------------
if study_mode == "Topic-wise Drill":
    st.title("🎯 UPSC Prelims MCQ Practice Drill")
    st.write("Generate authentic multi-statement prelims questions grounded in your uploaded syllabus materials.")

    col1, col2 = st.columns([3, 1])
    with col1:
        topic_input = st.text_input(
            "Enter syllabus topic or keyword:",
            placeholder="e.g., Doctrine of Basic Structure, BRICS Financial Architecture, Monetary Policy Committee"
        )
    with col2:
        q_count = st.selectbox("Number of Questions:", [3, 5, 10], index=0)

    if st.button("Generate Prelims Drill", type="primary"):
        if not topic_input.strip():
            st.warning("Please specify a topic.")
        else:
            with st.spinner("Searching Pinecone chunks and structuring UPSC questions..."):
                retrieved_chunks = retrieve_chunks(topic_input, top_k=5)
                quiz_data = generate_mcqs_from_context(topic_input, retrieved_chunks, num_questions=q_count)
                
                if quiz_data:
                    st.session_state.active_quiz = quiz_data
                    st.session_state.user_answers = {}
                    st.session_state.submitted = False
                    st.session_state.current_chunks = retrieved_chunks

    # Display active drill
    if st.session_state.active_quiz:
        st.markdown("---")
        with st.expander("📚 View Retrieved Source Chunks Used for this Test", expanded=False):
            if "current_chunks" in st.session_state and st.session_state.current_chunks:
                for idx, c in enumerate(st.session_state.current_chunks, 1):
                    st.markdown(f"**Chunk {idx}:**")
                    st.info(c)
            else:
                st.write("Using general GS syllabus grounding.")

        st.subheader("Prelims Examination Sheet")

        # Question rendering form
        with st.form("quiz_form"):
            for q in st.session_state.active_quiz:
                q_id = str(q.get("id", 1))
                st.markdown(f"#### Q{q_id}. {q.get('question_stem', '')}")
                
                # Render multi-statement block
                for stmt in q.get("statements", []):
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{stmt}")
                
                st.markdown(f"*{q.get('directive', 'Which of the statements given above is/are correct?')}*")
                
                options = q.get("options", [])
                st.session_state.user_answers[q_id] = st.radio(
                    label=f"Select answer for Q{q_id}:",
                    options=options,
                    index=None,
                    key=f"radio_{q_id}",
                    label_visibility="collapsed"
                )
                st.markdown("---")

            submit_btn = st.form_submit_button("Submit Answers & View Evaluation", type="primary")

        if submit_btn:
            st.session_state.submitted = True

        if st.session_state.submitted:
            correct_count = 0
            total_count = len(st.session_state.active_quiz)

            st.markdown("## 📊 Performance & Detailed Explanations")

            for q in st.session_state.active_quiz:
                q_id = str(q.get("id", 1))
                selected = st.session_state.user_answers.get(q_id)
                correct_letter = q.get("correct_option", "A").strip()
                
                # Determine if user selected correct option
                is_correct = selected and selected.startswith(correct_letter)
                if is_correct:
                    correct_count += 1
                    status_banner = "✅ Correct"
                elif selected is None:
                    status_banner = "⚠️ Unattempted"
                else:
                    status_banner = "❌ Incorrect"

                with st.container():
                    st.markdown(f"### Q{q_id} — {status_banner}")
                    st.markdown(f"**Your Choice:** {selected if selected else 'Not attempted'}")
                    st.markdown(f"**Official Answer:** Option {correct_letter}")
                    
                    st.markdown(f"**Detailed UPSC Evaluation:**")
                    st.success(q.get("explanation", "No detailed explanation provided."))
                    
                    if q.get("grounded_chunk_excerpt"):
                        st.caption(f"**Source Document Grounding:** *\"{q.get('grounded_chunk_excerpt')}\"*")
                    st.markdown("---")

            # Prelims Marking Scheme (+2 for correct, -0.66 for incorrect)
            unattempted_count = sum(1 for q in st.session_state.active_quiz if st.session_state.user_answers.get(str(q.get("id"))) is None)
            incorrect_count = total_count - correct_count - unattempted_count
            score = (correct_count * 2.0) - (incorrect_count * 0.66)

            st.metric(
                label="Prelims Score (UPSC GS Paper I standard)",
                value=f"{score:.2f} / {total_count * 2:.2f}",
                delta=f"{correct_count} Correct | {incorrect_count} Wrong | {unattempted_count} Left"
            )

# ---------------------------------------------------------
# View 2: Browse Document Chunks
# ---------------------------------------------------------
elif study_mode == "Browse Document Chunks":
    st.title("📖 Vector Store Chunk Inspector")
    st.write("Query Pinecone directly to see the exact text chunks indexed from your PDFs.")

    search_query = st.text_input("Inspect chunks for topic:", placeholder="e.g., Preamble amendment, Judicial Review")
    k_val = st.slider("Number of Chunks to inspect:", min_value=1, max_value=8, value=3)

    if st.button("Inspect Chunks", type="primary"):
        if search_query.strip():
            with st.spinner("Fetching nearest vector matches..."):
                chunks = retrieve_chunks(search_query, top_k=k_val)
                if chunks:
                    for i, chunk in enumerate(chunks, 1):
                        with st.expander(f"Chunk #{i} (Cosine Match)", expanded=True):
                            st.write(chunk)
                else:
                    st.info("No matching chunks found.")
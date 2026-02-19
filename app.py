
import os
import streamlit as st

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="EduAssist AI", layout="wide")

INDEX_PATH = "index"


# ----------------------------
# LOAD MODELS (CACHED)
# ----------------------------
@st.cache_resource
def load_models():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Summarization model (lighter, faster)
    summarizer_llm = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_new_tokens=200
    )

    # Q&A model (stronger reasoning)
    qa_llm = pipeline(
        "text-generation",
        model="google/flan-t5-large",
        max_new_tokens=256
    )

    return embeddings, summarizer_llm, qa_llm


embeddings, summarizer_llm, qa_llm = load_models()


# ----------------------------
# LOAD FAISS INDEX
# ----------------------------
@st.cache_resource
def load_vectorstore():

    if not (
        os.path.exists("index/index.faiss")
        and os.path.exists("index/index.pkl")
    ):
        raise RuntimeError(
            "FAISS index not found. Upload the index folder with index.faiss and index.pkl"
        )

    return FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# ----------------------------
# SUMMARIZATION
# ----------------------------
def summarize_text(text):

    prompt = f"""
Summarize the following research content into concise study notes.

Rules:
- Bullet points
- Clear academic language
- No repetition
- No extra commentary

Text:
{text}
"""

    output = summarizer_llm(prompt)[0]["generated_text"]
    return output.strip()


# ----------------------------
# QUESTION ANSWERING (RAG)
# ----------------------------
def answer_question(question):

    docs = retriever.invoke(question)

    if not docs:
        return "Not found in the provided material."

    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

Rules:
- If answer is not present in context, respond:
  "Not found in the provided material."
- Be concise
- Do not hallucinate
"""

    output = qa_llm(prompt)[0]["generated_text"]
    return output.strip()


# ----------------------------
# STREAMLIT UI
# ----------------------------
st.title("📚 ScholarAI — Research Paper Assistant")

tab1, tab2 = st.tabs(["📝 Summarization", "❓ Q&A"])


# ----------------------------
# TAB 1: SUMMARIZER
# ----------------------------
with tab1:
    st.subheader("Summarize Research Content")

    input_text = st.text_area("Paste paper content", height=250)

    if st.button("Summarize"):
        if not input_text.strip():
            st.warning("Please provide text.")
        else:
            with st.spinner("Summarizing..."):
                result = summarize_text(input_text)
                st.success(result)


# ----------------------------
# TAB 2: Q&A
# ----------------------------
with tab2:
    st.subheader("Ask Questions (RAG)")

    question = st.text_input("Enter your question")

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching knowledge base..."):
                result = answer_question(question)
                st.success(result)

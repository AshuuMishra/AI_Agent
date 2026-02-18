
import os
import streamlit as st

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="ScholarAI", layout="wide")

INDEX_PATH = "index"   # index.faiss + index.pkl must exist


# ----------------------------
# LOAD MODELS (ONCE)
# ----------------------------
@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    llm = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256,
        do_sample=False
    )

    return embeddings, llm


embeddings, llm = load_models()


# ----------------------------
# LOAD VECTOR STORE (ONLY LOAD)
# ----------------------------
@st.cache_resource
def load_vectorstore():
    if not (
        os.path.exists("index.faiss")
        and os.path.exists("index.pkl")
    ):
        raise RuntimeError(
            "FAISS index not found. index.faiss and index.pkl must exist."
        )

    return FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# ----------------------------
# CORE FUNCTIONS
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
    output = llm(prompt)[0]["generated_text"]
    return output.strip()


def answer_question(question):
    docs = retriever.invoke(question)

    if not docs:
        return "Not found in the provided material."

    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
You are answering strictly from the given context.

Context:
{context}

Question:
{question}

Rules:
- If the answer is not in the context, say:
  "Not found in the provided material."
- Be concise and factual.
"""

    output = llm(prompt)[0]["generated_text"]
    return output.strip()


# ----------------------------
# STREAMLIT UI
# ----------------------------
st.title("📚 ScholarAI — Research Paper Assistant")

tab1, tab2 = st.tabs(["📝 Summarization", "❓ Q&A"])

with tab1:
    st.subheader("Summarize Research Content")

    input_text = st.text_area(
        "Paste paper content",
        height=250
    )

    if st.button("Summarize"):
        if not input_text.strip():
            st.warning("Please provide text.")
        else:
            with st.spinner("Summarizing..."):
                st.success(summarize_text(input_text))


with tab2:
    st.subheader("Ask Questions (RAG)")

    question = st.text_input("Enter your question")

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching knowledge base..."):
                st.success(answer_question(question))


import os
import streamlit as st

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline


st.set_page_config(page_title="EduAssist AI", layout="wide")

INDEX_PATH = "index"


# ----------------------------
# LOAD MODELS
# ----------------------------
@st.cache_resource
def load_models():
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Summarizer (light)
        summarizer_llm = pipeline(
            "text-generation",
            model="google/flan-t5-base",
            max_new_tokens=200
        )

        # Q&A model
        qa_llm = pipeline(
            "text-generation",
            model="google/flan-t5-large",
            max_new_tokens=256
        )

        return embeddings, summarizer_llm, qa_llm

    except Exception as e:
        st.error("Model loading failed.")
        st.exception(e)
        st.stop()


embeddings, summarizer_llm, qa_llm = load_models()


# ----------------------------
# LOAD VECTOR STORE
# ----------------------------
@st.cache_resource
def load_vectorstore():
    try:
        if not (
            os.path.exists("index/index.faiss")
            and os.path.exists("index/index.pkl")
        ):
            raise RuntimeError("FAISS index missing")

        return FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    except Exception as e:
        st.error("Vectorstore loading failed.")
        st.exception(e)
        st.stop()


vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})


# ----------------------------
# SUMMARIZATION
# ----------------------------
def summarize_text(text):
    prompt = f"""
Summarize the following research content into concise study notes.

Text:
{text}
"""
    result = summarizer_llm(prompt)[0]["generated_text"]
    return result.strip()


# ----------------------------
# Q&A
# ----------------------------
def answer_question(question):

    docs = retriever.invoke(question)

    if not docs:
        return "Not found in the provided material."

    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
Answer using ONLY the context.

Context:
{context}

Question:
{question}
"""

    result = qa_llm(prompt)[0]["generated_text"]
    return result.strip()


# ----------------------------
# UI
# ----------------------------
st.title("📚 ScholarAI — Research Paper Assistant")

tab1, tab2 = st.tabs(["Summarization", "Q&A"])


with tab1:
    text = st.text_area("Paste research content")

    if st.button("Summarize"):
        st.write(summarize_text(text))


with tab2:
    q = st.text_input("Ask a question")

    if st.button("Get Answer"):
        st.write(answer_question(q))

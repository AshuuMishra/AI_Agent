📚 EduAssist AI — Intelligent AI Agent for Personalized Learning

EduAssist AI is a Generative AI–powered academic assistant designed to help students learn more effectively using their own study materials.

Built with modern GenAI technologies such as LangChain,Langsmith,HuggingFace vector search, and Retrieval-Augmented Generation (RAG), EduAssist AI provides personalized, grounded, and interactive academic support.


🎯 Key Capabilities

EduAssist AI helps students by:

✅ Summarizing research papers into concise study notes
✅ Answering personalized questions based on uploaded content
✅ Providing context-aware responses using RAG
✅ Supporting interactive academic exploration

🧠 GenAI Features Implemented

This project demonstrates practical usage of several advanced Generative AI concepts:

✨ 1. Few-Shot Prompting

Custom prompt templates guide the AI to:

Generate structured summaries

Provide concise factual answers

📄 2. Document Understanding

Academic documents are:

Split into manageable chunks

Processed using LangChain document 

Converted into vector embeddings

🔎 3. Embeddings

Text chunks are transformed into numerical representations using:
HuggingFace AI Embeddings

Sentence-Transformer models

This enables semantic understanding of academic content.

⚡ 4. Retrieval-Augmented Generation (RAG)

The core architecture:

User asks a question

System retrieves relevant document chunks

AI generates an answer strictly grounded in context

This ensures:

Higher accuracy

Reduced hallucinations

Personalized responses

🗄️ 5. Vector Search & Database

ScholarAI uses vector storage for similarity search:

FAISS vector store

In-memory retrieval system

Semantic search for academic content

🏗️ Tech Stack
Category	Tools
LLM	FLAN-T5-Base
Framework	LangChain
Embeddings	HUggineFaceEmbeddings / SentenceTransformers
Vector DB	FAISS
UI	Streamlit
Language	Python
📂 Project Structure
ScholarAI/
│── app.py
│── requirements.txt
│── index/
│    ├── index.faiss
│    └── index.pkl
│── notebook.ipynb
│── README.md

⚙️ How It Works
Step 1 — Document Processing

Research papers are:

Split into chunks

Embedded into vectors

Stored in FAISS

Step 2 — Question Answering

When a user asks a question:

Relevant chunks are retrieved

Context is passed to the LLM

Answer is generated based only on retrieved data

Step 3 — Summarization

Users can paste any academic text and get:

Clean bullet-point notes

Concise summaries

Study-ready output

💻 Installation
1️⃣ Clone the repo
git clone https://github.com/AshuuMishra/AI_Agent
cd ScholarAI

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the app
streamlit run app.py

🌟 Use Cases

EduAssist AI can help:

🎓 Students studying research papers
📚 Researchers summarizing literature
🧑‍🏫 Teachers preparing notes
💡 Self-learners understanding complex topics

📈 Future Improvements

Multi-PDF upload support

Chat history memory

Citation generation

Voice interaction

Cloud deployment optimization

🙌 Acknowledgements

Special thanks to:

Google GenAI Team

Kaggle Learning Platform

LangChain Community

📜 License

This project is open-source and available under the MIT License.

⭐ Support

If you find this project helpful:

👉 Star the repo
👉 Share with others
👉 Contribute improvements

# 🧠 Self-RAG Chatbot

A smart chatbot that reads your PDFs and answers questions using Self-RAG architecture with web search fallback.

## 🔗 Live Demo
[Try it here](https://your-app.streamlit.app)

## ✨ Features
- 📄 Multi-PDF upload & chat
- 🧠 Self-RAG (auto decides retrieval)
- 🌐 Web search fallback (Tavily)
- 💬 Multiple chat sessions
- 🔐 Persistent memory (MongoDB)
- 🔑 Unique user sessions

## 🛠️ Tech Stack
- LangGraph + LangChain
- Groq (Llama 3.3 70B)
- FAISS + HuggingFace Embeddings
- Tavily Web Search
- MongoDB Atlas
- Streamlit

## ⚙️ Setup

```bash
git clone https://github.com/yourusername/self-rag.git
cd self-rag
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env`:
```env
GROQ_API_KEY=your_key
TAVILY_API_KEY=your_key
MONGODB_URI=your_uri
HF_TOKEN=your_key
```

Run:
```bash
streamlit run app.py
```

## 👩‍💻 Developer
**Anamika Kumari** — B.Tech CS, Dumka Engineering College  
Data Science & AI/ML Intern @ Ardent Computech Pvt. Ltd.

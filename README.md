# 🧬 AI Research Protocol Assistant

<div align="center">
  <p>An intelligent, protocol-aware laboratory assistant powered by Agentic Workflows, Dual-LLM architecture, and Advanced RAG.</p>
</div>

---

## 🌟 Overview

The **AI Research Protocol Assistant** is a full-stack, AI-powered conversational agent designed to help scientists, researchers, and lab technicians manage, query, and troubleshoot complex laboratory protocols. 

By leveraging **Advanced Retrieval-Augmented Generation (RAG)** combined with **Agentic Workflows**, this assistant doesn't just answer questions—it understands context, pulls relevant data from uploaded PDFs, chunks and vectorizes complex technical data, and assists in step-by-step problem-solving.

## ✨ Key Features

- 🧠 **Agentic Workflows (LangGraph)**: Uses intelligent nodes and edges to classify queries, route tasks to specialized agents (e.g., specific protocol extraction vs. general Q&A), and synthesize multi-step reasoning.
- 🚀 **Dual-LLM Architecture**: 
  - **Gemini (Google Generative AI)**: Handles complex reasoning, retrieval synthesis, and conversational memory.
  - **Groq (Llama 3)**: Handles high-RPM, low-latency background tasks like metadata extraction, document summarization, and high-volume chunking.
- 📚 **Advanced RAG (Hybrid Search)**: Combines dense vector retrieval via **Pinecone** (powered by `GoogleGenerativeAIEmbeddings`) with sparse retrieval (`BM25`) for highly accurate protocol lookups.
- 📄 **Robust PDF Parsing**: Uses `pdfplumber` to accurately extract both complex text and tabular data from raw laboratory documents.
- 💾 **Long-term Memory**: Integrates **Redis** to persist conversational history and context across user sessions.
- ⚡ **Real-time WebSockets**: Provides a fluid, real-time chat interface without latency overhead, handling secure JWT authentication securely over standard WebSocket connections.
- ☁️ **Cloud-Native**: Optimized memory footprint (< 512MB RAM) for seamless, cost-effective deployments on platforms like **Render**.

---

## 🛠️ Technology Stack

### Backend
* **Framework**: FastAPI (Python 3.11+)
* **AI Orchestration**: LangChain & LangGraph
* **LLMs**: Google Gemini API & Groq API
* **Vector Store**: Pinecone
* **Database**: PostgreSQL (via SQLAlchemy / asyncpg)
* **Caching & Memory**: Redis
* **PDF Extraction**: pdfplumber, pypdf

### Frontend
* **Framework**: React / Vite (TypeScript)
* **Styling**: TailwindCSS
* **State Management**: Context API
* **Communication**: REST APIs & Native WebSockets

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Node.js 18+
- PostgreSQL (Local or Cloud)
- Redis Server
- API Keys: Google Gemini, Groq, Pinecone

### 1. Clone the Repository
```bash
git clone https://github.com/SagnikDey1503/protocol-checker.git
cd protocol-checker
```

### 2. Backend Setup
```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Setup environment variables
cp .env.example .env
```

**Configure your `.env` file** with the required keys:
```env
# LLM Providers
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key

# Vector DB
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=protocol-assistant

# Database & Cache
DATABASE_URL=postgresql+asyncpg://user:password@localhost/dbname
REDIS_URL=redis://localhost:6379

# Auth
JWT_SECRET_KEY=your_super_secret_key
```

**Run Database Migrations & Start Server:**
```bash
alembic upgrade head
fastapi run app/main.py --reload
```
The backend API will be running at `http://localhost:8000`.

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
The frontend UI will be running at `http://localhost:5173`.

---

## ☁️ Deployment

This project is fully configured for deployment on **Render**. It includes a `render.yaml` Blueprint which automatically provisions:
1. **Web Service**: The FastAPI backend (Dockerized)
2. **PostgreSQL Database**: Managed relational database
3. **Redis Instance**: Managed cache/memory server
4. **Static Site**: The React frontend

**To Deploy:**
1. Connect your GitHub repository to Render.
2. Under "Blueprints", select `render.yaml`.
3. Provide the required Environment Variables in the Render Dashboard.
4. Render will automatically build and deploy all 4 services.

*(Note: The backend Dockerfile is heavily optimized to run well within Render's 512MB Free Tier limits).*

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the app, add more parsing strategies, or refine the Agentic logic, feel free to open a Pull Request.

## 📝 License

This project is licensed under the MIT License.

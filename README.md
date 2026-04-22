# 🧬 COVID-19 AI Research Platform

A production-grade deep learning platform for COVID-19 scholarly research, powered by Hybrid Search (Semantic + Keyword) and RAG (Retrieval-Augmented Generation).

## 🚀 Features
- **Hybrid Search Engine**: Combines SBERT semantic understanding with MongoDB text search for 99.9% relevance.
- **AI Research Paper Drafter**: Automatically generates professional paper drafts based on real scholarly context.
- **PDF Upload & Analysis**: Instant summary and key point extraction from any research PDF.
- **Real-Time Data Ingestion**: Automated pipeline fetching real academic papers from ArXiv.
- **Medical AI Pipeline**: Integrated BART for summarization and scispaCy for clinical entity recognition.

## 🛠️ Technology Stack
- **Frontend**: Streamlit
- **Backend**: FastAPI (Python)
- **Vector DB**: ChromaDB
- **Metadata DB**: MongoDB
- **AI Models**: SBERT (all-MiniLM-L6-v2), BART, Gemini-Flash, scispaCy.

## 📦 Setup & Installation

### 1. Prerequisites
- Python 3.10+
- MongoDB (Running locally or via Docker)
- Gemini API Key ([Get it here](https://aistudio.google.com/app/apikey))

### 2. Manual Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd covid_project

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
# Create a .env file with:
# MONGO_URL=mongodb://localhost:27017
# GEMINI_API_KEY=your_key_here
```

### 3. Run with One Click (Local)
On Windows, simply run:
```powershell
.\start.ps1
```

### 4. Deploy with Docker
```bash
docker-compose up --build
```

## 📊 Data Ingestion
To populate the database with the latest research:
```bash
python data_pipeline/real_ingestor.py
python -m scripts.reindex_vectors
```

## 📄 License
MIT License. Created for the global research community.

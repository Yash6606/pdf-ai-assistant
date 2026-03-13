# 📚 AI PDF Assistant (Advanced RAG with OCR)

An intelligent **PDF Question-Answering system** built using **Retrieval Augmented Generation (RAG)**.
The system can read both **text PDFs and scanned PDFs**, retrieve relevant context using **hybrid search**, and generate accurate answers using a **Large Language Model (LLM)**.

---

# 🚀 Features

✅ Ask questions from any PDF
✅ Supports **scanned PDFs using OCR**
✅ **Semantic chunking** for better document understanding
✅ **Hybrid retrieval** (Vector Search + BM25 keyword search)
✅ **Cross-encoder reranking** for improved answer relevance
✅ **Local embedding storage (FAISS)** to avoid recomputation
✅ **Chat history memory** for conversational interaction
✅ Source context display for transparency

---

# 🧠 System Architecture

The system uses an **Advanced Retrieval Augmented Generation (RAG)** pipeline.

PDF
↓
OCR (for scanned documents)
↓
Semantic Chunking
↓
Embedding Generation
↓
FAISS Vector Database (stored locally)
↓
Hybrid Search (Vector + BM25)
↓
Cross-Encoder Reranker
↓
LLM (DeepSeek via OpenRouter)
↓
Answer Generation with Source Context

---

# 📦 Tech Stack

**Language**

Python

**Framework**

Streamlit

**Libraries**

LangChain
Sentence Transformers
FAISS
Rank-BM25
PyMuPDF
Pytesseract

**Models**

Embedding Model
sentence-transformers/all-MiniLM-L6-v2

Reranker
cross-encoder/ms-marco-MiniLM-L-6-v2

LLM
DeepSeek Chat (via OpenRouter)

---

# 📁 Project Structure

```
project/
│
├── app.py
├── requirements.txt
├── README.md
│
├── vectorstore_<pdf_name>/
│   ├── index.faiss
│   ├── index.pkl
│
└── temp_pdf_files/
```

---

# ⚙️ Installation

### 1️⃣ Clone the repository

```
git clone https://github.com/yourusername/pdf-ai-assistant.git
cd pdf-ai-assistant
```

### 2️⃣ Create a virtual environment

```
python -m venv rag_env
```

Activate it

Windows

```
rag_env\Scripts\activate
```

Linux / Mac

```
source rag_env/bin/activate
```

---

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

### 4️⃣ Install Tesseract OCR

Download and install:

https://github.com/UB-Mannheim/tesseract/wiki

Example path:

```
C:\Program Files\Tesseract-OCR\tesseract.exe
```

---

# 🔑 Setup API Key

Create an API key from **OpenRouter**.

https://openrouter.ai

Then set it in your environment:

Windows

```
set OPENAI_API_KEY=your_key_here
```

Linux / Mac

```
export OPENAI_API_KEY=your_key_here
```

---

# ▶ Run the Application

```
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

# 📊 Example Workflow

1. Upload a PDF document
2. The system extracts text and images
3. OCR processes scanned content
4. Text is chunked and converted into embeddings
5. FAISS stores embeddings locally
6. Hybrid search retrieves relevant chunks
7. Reranker selects best context
8. LLM generates an answer

---

# 🧩 Future Improvements

Possible upgrades for a production-level system:

• Multi-PDF knowledge base
• Query rewriting for better retrieval
• Context compression
• Knowledge graph retrieval (GraphRAG)
• Streaming LLM responses
• Document highlighting in PDF

---

# 📚 Use Cases

Academic paper assistant
Research document exploration
Company knowledge base
Legal document analysis
AI study assistant

---

# ⭐ Advanced RAG Techniques Used

Semantic chunking
Hybrid retrieval
Cross-encoder reranking
Embedding persistence
OCR document understanding

---

# 👨‍💻 Author

Developed by **Yash Patel**

AI / ML Engineer
Focused on **RAG systems, LLM applications, and intelligent document processing**

---

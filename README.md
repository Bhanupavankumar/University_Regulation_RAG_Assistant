# 🎓 University Regulation RAG Assistant
### *A Retrieval-Augmented Generation (RAG) System for University Academic Bylaws & Regulations*

<div align="center">

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-university--rag--assistant.netlify.app-00C7B7?style=for-the-badge&logo=netlify&logoColor=white)](https://university-rag-assistant.netlify.app)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Bhanupavankumar/University_Regulation_RAG_Assistant)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React 19](https://img.shields.io/badge/React-19.0-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-FF6F00?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![Groq LLM](https://img.shields.io/badge/LLM-Groq_Inference-F55036?style=for-the-badge)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

### 🚀 **[👉 Click Here to Open Live Application](https://university-rag-assistant.netlify.app)**

[🌐 Live Demo](https://university-rag-assistant.netlify.app) • [📖 Abstract](#1-abstract) • [🎯 Objectives](#3-objectives) • [🛠️ Technologies](#4-technologies-used) • [🔄 Methodology](#5-methodology) • [⚙️ Implementation](#6-implementation) • [📊 Results & Tests](#7-results-and-outputs) • [⚡ Quick Start](#-quick-start-local-development)

---

</div>

## 1. Abstract

The **University Regulation RAG Assistant** is a Retrieval-Augmented Generation (RAG) based question-answering system designed to provide accurate answers from university documents. The system works with documents such as university regulations, admission policies, scholarship policies, examination guidelines, attendance rules, and academic handbooks.

Multiple PDF documents can be uploaded to the system. The documents are loaded and divided into smaller chunks. These chunks are converted into numerical representations called embeddings and stored in a **FAISS** vector database. When a user asks a question, the system searches the vector database to find the most relevant sections of the uploaded documents. These sections are then provided as context to a **Groq Large Language Model (LLM)**, which generates a concise answer based only on the retrieved information.

The system also displays the source document and page number used to generate the answer. If the required information is not available in the uploaded documents, the system informs the user that the answer could not be found instead of generating unsupported information.

---

## 2. Problem Statement

University regulations and academic information are usually available across multiple PDF documents. Students and faculty find it difficult and time-consuming to manually search through these dense documents to find specific information about attendance, examinations, admission, scholarships, academic rules, and other policies.

Therefore, there is a strong need for an intelligent assistant that can:
- 📄 Accept multiple university PDF documents.
- ⚙️ Understand and preprocess the document content.
- 🔍 Retrieve relevant information for a user's question.
- 🎯 Generate answers based **only** on the provided documents.
- 🏷️ Show the exact **source document and page number**.
- 🚫 Avoid generating unsupported answers when the required information is not available.

The **University Regulation RAG Assistant** solves this problem using Retrieval-Augmented Generation.

---

## 3. Objectives

The main objectives of the project are:
1. **Develop a RAG-based assistant** for university regulations.
2. **Support multiple PDF document uploads** simultaneously.
3. **Extract text** accurately from university PDF documents.
4. **Divide large documents** into smaller meaningful chunks using recursive splitting.
5. **Convert document chunks into embeddings** using semantic models.
6. **Store embeddings in a FAISS vector database** for rapid similarity search.
7. **Retrieve relevant document sections** based on user queries.
8. **Generate answers using a Large Language Model (Groq LLM)**.
9. **Ground the generated answers** strictly in the retrieved documents.
10. **Display the source document and page number** with every citation.
11. **Reduce hallucination** by restricting the LLM to the retrieved context.
12. **Indicate clearly when the requested information cannot be found** in the uploaded documents.

---

## 4. Technologies Used

| Technology | Purpose |
| :--- | :--- |
| **Python** | Main programming language for the RAG pipeline and server |
| **Google Colab** | Interactive development, experimentation, and execution environment |
| **PyPDF** | Loading and extracting text from multi-page PDF files |
| **LangChain** | RAG pipeline orchestrator and document processing utilities |
| **RecursiveCharacterTextSplitter** | Splitting documents into sliding window chunks (800 chars / 150 overlap) |
| **Hugging Face / Sentence Transformers** | Embedding generation using `sentence-transformers/all-MiniLM-L6-v2` |
| **FAISS (Facebook AI Similarity Search)** | Vector storage and high-speed similarity search |
| **Groq Cloud** | High-speed LLM inference |
| **FastAPI & Uvicorn** | High-performance asynchronous REST API backend |
| **React 19 & Vite** | Modern reactive user interface with Glassmorphism styling |
| **NumPy** | High-performance numerical operations and matrix similarity calculations |

### Core AI Components
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors)
- **Vector Database**: `FAISS (IndexFlatIP / Cosine Distance)`
- **LLM**: `Groq API` (e.g. `llama-3.3-70b-versatile`, `openai/gpt-oss-20b`)

---

## 5. Methodology

```
┌──────────────────────────────────────────────────┐
│              Uploaded PDF Documents              │
└────────────────────────┬─────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────┐
│        Document Loading (PyPDF / Text)           │
└────────────────────────┬─────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────┐
│                 Text Extraction                  │
└────────────────────────┬─────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────┐
│  Document Chunking (Size: 800, Overlap: 150)     │
└────────────────────────┬─────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────┐
│   Text Embeddings (all-MiniLM-L6-v2 - 384 dim)   │
└────────────────────────┬─────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────┐
│              FAISS Vector Database               │
└────────────────────────┬─────────────────────────┘
                         │
        User Query ──────┼─────────────────────────┐
                         ▼                         ▼
                  Query Embedding          Similarity Search
                         │                         │
                         └────────────┬────────────┘
                                      ▼
┌──────────────────────────────────────────────────┐
│             Relevant Document Chunks             │
└────────────────────────┬─────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────┐
│    Context Construction (Grounded Prompting)     │
└────────────────────────┬─────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────┐
│                    Groq LLM                      │
└────────────────────────┬─────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────┐
│     Grounded Answer + Source & Page Number       │
└──────────────────────────────────────────────────┘
```

---

## 6. Implementation

### Step 1: Upload Documents
The system allows multiple PDF documents to be uploaded. The project works with official regulations such as:
- *University Research Policy*
- *Admission Policy and Procedure*
- *Scholarships Policy*
- *R22.1 Regulations for B.Tech*

### Step 2: Document Loading
Uploaded PDF documents are loaded using `PyPDFLoader`. Each page is treated as a document object containing:
- Page content
- Source filename (`source`)
- Page number (`page_number`)
- Category & section metadata

*Example Metadata:*
```json
{
  "source": "R22.1 Regulations for B.Tech.pdf",
  "page_number": 21,
  "category": "Academic"
}
```

### Step 3: Document Chunking
Large documents cannot be passed to the LLM as one single block. Documents are split into semantic chunks using `RecursiveCharacterTextSplitter`:
- **Chunk Size**: `800` characters
- **Chunk Overlap**: `150` characters
- **Separators**: `["\n\n", "\n", ". ", "; ", ", ", " ", ""]` (splits on paragraphs, sentences, words).

*Why chunking is required:* Chunking optimizes retrieval precision, prevents context window overflow, and enables pinpoint source citation.

### Step 4: Embeddings
Each chunk is transformed into a 384-dimensional dense vector using `sentence-transformers/all-MiniLM-L6-v2`. This captures the conceptual and semantic meaning of the policy text:
- Query: *"What is the minimum attendance requirement?"*
- Document chunk: *"The attendance in each course shall not be less than 75%..."*
- Both map to proximate positions in the vector space, enabling semantic retrieval.

### Step 5: FAISS Vector Search
Embeddings are indexed in FAISS. When a user asks a question:
1. The question is converted into an embedding.
2. FAISS performs an inner product / cosine distance calculation against all indexed chunks.
3. The top-$k$ most similar chunks are retrieved.

### Step 6: Metadata Handling & Deduplication
The system preserves metadata (`source`, `page_number`) alongside every vector chunk. Before assembling the final prompt context, duplicate source-page combinations are removed to maximize context density.

### Step 7: Retrieval Process
When a query is received:
1. Receives the user question.
2. Converts question into an embedding vector.
3. Searches FAISS vector database with $k = 10$.
4. Applies distance filtering (`if score < 1.2`).
5. Deduplicates source and page combinations.
6. Builds a clean context block.
7. Dispatches context and question to Groq LLM.

### Step 8: Grounded Answer Generation
The retrieved chunks are formatted into a constrained prompt sent to the **Groq LLM**:
- Instructs model to **use only the provided documents**.
- Prohibits outside knowledge or policy fabrication.
- Requires concise, direct answers with clause citations.
- **Strict Fallback Guarantee**: If the answer cannot be found in the retrieved chunks, the model outputs:
  > *"The answer could not be found in the uploaded university documents."*

---

## 7. Results and Outputs

The system was evaluated against real-world academic regulatory test cases:

### 🧪 Test Case 1: Attendance Rule
- **User Question**: `What is the minimum attendance requirement?`
- **Generated Answer**: `The minimum attendance requirement is 75%.`
- **Source**: `R22.1 Regulations for B.Tech` — **Page 21**
- **Outcome**: ✅ *Accurately retrieved minimum threshold and source page.*

---

### 🧪 Test Case 2: Practical Examination
- **User Question**: `How is the end semester examination conducted for practical courses?`
- **Generated Answer**: `The practical examination is conducted jointly by two examiners, with the assessment scheme communicated by the laboratory in-charge and the summative assessment conducted for a maximum of 40 marks.`
- **Sources**: `R22.1 Regulations for B.Tech` — **Pages 8, 22, 24, 29**
- **Outcome**: ✅ *Correctly synthesized multi-page policy criteria.*

---

### 🧪 Test Case 3: Supplementary Examination
- **User Question**: `How often are supplementary examinations conducted?`
- **Generated Answer**: `Supplementary examinations are conducted once, during the summer semester.`
- **Source**: `R22.1 Regulations for B.Tech` — **Page 38**
- **Outcome**: ✅ *Direct citation of summer semester ordinance.*

---

### 🧪 Test Case 4: Information Not Available (Hallucination Prevention)
- **User Question**: `What is the hostel mess fee for this semester?`
- **Generated Answer**: `The answer could not be found in the uploaded university documents.`
- **Outcome**: ✅ *Zero hallucination; properly triggered fallback response when information was absent.*

---

## 8. Advantages

1. **Easy Access to University Information**: Instant answers eliminate tedious manual searching through 50+ page PDFs.
2. **Multiple Document Support**: Ingests handbooks, examination codes, research bylaws, and admission guidelines simultaneously.
3. **Semantic Search**: FAISS retrieves answers based on contextual meaning rather than simple keyword matching.
4. **Verifiable Source Attribution**: Displays exact document name and page number for every statement.
5. **Hallucination Guardrails**: The LLM is strictly constrained to retrieved context with negative fallback triggers.

---

## 9. Limitations

1. **Document Quality Dependency**: Scanned or low-resolution PDFs may require OCR preprocessing for clean text extraction.
2. **Chunking Trade-offs**: Sub-optimal chunk boundaries can occasionally split related clauses across two chunks.
3. **Embedding Model Specifics**: Semantic accuracy depends on the vocabulary and domain coverage of the embedding model.
4. **Duplicate Uploads**: Uploading identical PDFs under different filenames creates duplicate context candidates.

---

## 10. Conclusion

The **University Regulation RAG Assistant** demonstrates how Retrieval-Augmented Generation can be practically applied to institutional academic bylaws and policies.

By combining multi-PDF text extraction, recursive chunking, `all-MiniLM-L6-v2` embeddings, FAISS vector search, and Groq LLM synthesis, the platform delivers grounded, concise, and verifiable answers with source document and page number citations. The integrated fallback mechanism ensures reliable academic compliance and eliminates hallucinations.

---

## 🚀 Quick Start (Local Development)

### Prerequisites
- **Python**: 3.10+
- **Node.js**: 18+

### 1. Backend Setup
```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # On Windows
# source venv/bin/activate    # On Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Seed regulations database
python seed_data.py

# Start FastAPI server
python run.py
```
> Backend runs at: `http://127.0.0.1:8000` | Swagger Docs: `http://127.0.0.1:8000/docs`

### 2. Frontend Setup
```bash
# In a new terminal
cd frontend
npm install
npm run dev
```
> Frontend runs at: `http://localhost:5173`

---

## 📓 Google Colab Pipeline

For interactive experimentation with RAG chunking, FAISS index construction, and prompt experiments, open [`University_RAG_Colab_Pipeline.ipynb`](./University_RAG_Colab_Pipeline.ipynb) directly in Google Colab.

---

## 📄 License

Distributed under the [MIT License](LICENSE).  
Developed by **Bhanu Pavan Kumar**.

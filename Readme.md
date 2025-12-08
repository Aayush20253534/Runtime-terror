<center><h1>Neuro Ranger</h1>
 
## An AI-Powered RAG Knowledge Engine
[![Live Demo](https://img.shields.io/badge/Neuro--Ranger-Live%20Demo-brightgreen?style=for-the-badge&logo=github)](https://shreyansh-kushw.github.io/Neuro-Ranger/)  

</center>

A smart Retrieval-Augmented Generation (RAG) system that processes documents, performs OCR, creates concise summaries, and enables high-accuracy semantic search across large unstructured datasets.

## 📘 Project Overview
Neuro Ranger acts as a personal “Second Brain,” ingesting PDFs and images, extracting text with OCR, generating AI-driven summaries, and powering natural-language search through vector-based semantic retrieval.

## 🚨 Problem Statement (PS 4)

The Yellow Ranger faces critical information overload from thousands of unorganized mission reports, manuals, and enemy intel. Manual processing is slow and impractical, causing delays and missed insights.

## ⭐ Features Implemented
- 📄 PDF & Image Upload + OCR Extraction
- 🤖 AI Summaries powered by Gemini API
- 🔎 Semantic Search using FAISS + Sentence Transformers
- 🗂️ JSON-based metadata tracking and data storage
- 🎨 Frontend using HTML + TailwindCSS + JS
- ⚡ Backend powered by FastAPI

## 🛠️ Tech Stack
<center>

### 🔹Backend
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)

### 🎨 Frontend
![HTML5](https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript&logoColor=black)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-06B6D4?logo=tailwindcss&logoColor=white)

### 🤖 AI / ML Tools
![Gemini](https://img.shields.io/badge/Gemini_API-4285F4?logo=google&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-2D3E50?logo=facebook&logoColor=white)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-1E90FF?logo=python&logoColor=white)

### 📄 OCR & Text Extraction
![PyMuPDF](https://img.shields.io/badge/PyMuPDF-008000?logo=python&logoColor=white)
![EasyOCR](https://img.shields.io/badge/EasyOCR-FFDD00?logo=python&logoColor=black)

</center>

## 🏗️ System Architecture

1. **User Interface (Frontend)**
   - HTML, JavaScript, TailwindCSS
   - Sends queries, files (PDFs/Images) to backend

2. **FastAPI Backend**
   - Handles API requests
   - Passes the requests made by the frontend to the backend

3. **OCR & Text Extraction Layer**
   - **PyMuPDF** for PDF text extraction
   - **EasyOCR** for image-based text extraction

4. **AI Embeddings Generator**
   - **Sentence-Transformers** for generating high-quality vector embeddings

5. **Vector Store**
   - **FAISS** for fast similarity search
   - Stores and retrieves embeddings

6. **Semantic Search Engine**
   - Finds top-k relevant text chunks
   - The relevant results are ranked using a CrossEncoder

7. **Response Generation**
   - **Gemini API** used for reasoning + answer formulation
   - Sends structured response back to frontend


## 📡 API Documentation
### 📤 **POST /preprocess**
Uploads a PDF/image file, performs OCR, extracts text, generates a summary,  
and indexes the content into the RAG system.

**Request:**  
- `file`: UploadFile (PDF/Image)

**Response:**  
- Processing status  
- Confirmation of ingestion into vector store  


### 📄 **GET /summary**
Returns the most recently generated summary along with metadata.

**Response:**  
- `summary`: Extracted summary  
- `category`: Document type (if detected)  
- `date`: Timestamp  


### ❓ **GET /ask**
Performs semantic search on the indexed content and generates an AI answer.

**Query Parameter:**  
- `query`: User question

**Response:**  
- Retrieved text chunks  
- Generated answer  
- Original user query


## 🧪 Setup Instructions
```bash
git clone https://github.com/Aayush20253534/Runtime-terror.git 
cd Runtim-terror/Backend

pip install -r requirements.txt
uvicorn main:app --reload  

```

## 🚀 Deployment Link
[![Live Demo](https://img.shields.io/badge/Neuro--Ranger-Live%20Demo-brightgreen?style=for-the-badge&logo=github)](https://shreyansh-kushw.github.io/Neuro-Ranger/)

## 🖼️ Screenshots
![Home Page](/assests/home.png)
![Summary Page](/assests/summary.png)
![History Page](/assests/history.png)

## ⚠️ Error Handling
- OCR fallback
- Structured API errors
- Retry logic for vector indexing

## 🤖 AI/ML Integration
- Gemini for summaries + Question answering  
- FAISS/sentence_transformers for embeddings  
- OCR via pymupdf/EasyOCR

## 👥 Team Members & Responsibilities
**Shreyansh Kushwaha** - Backend developement, improvement on semantic search and readme.  
**Aayush Thakur** - UI/UX and frontend designing and deployement  
**Prateek Rastogi** - Semantic search implementation  
**Aayansh Niranjan** - Integrating the whole project using **FastAPI**  and Readme creation

## 🔮 Future Improvements
- Multi-document reasoning, analysis and comparison
- Page wise summarization
- Advanced analytics
- Multi-documents questions and answers

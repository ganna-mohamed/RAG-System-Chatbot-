# 📖 PDF Q&A System (Transformers + LangChain + Streamlit)

## 🔹 Project Overview
This project is a *Question Answering system* that allows users to upload a PDF (e.g., research papers such as "Attention Is All You Need") and ask questions about its content.  
The system retrieves the most relevant text chunks from the document and generates answers using a Transformer model.

## 🔹 Features
- Upload any PDF file.
- Extract and split text into chunks.
- Convert text into embeddings with HuggingFace.
- Store and retrieve embeddings using FAISS.
- Answer questions interactively via a Streamlit app.
- Runs on *CPU* (no GPU required).

## 🔹 Technologies Used
- *Python* (3.13 / or recommended 3.11 for stability)
- *Streamlit* – interactive web app
- *LangChain* – for text splitting, embeddings, and retrieval
- *Transformers (HuggingFace)* – for the language model
- *FAISS* – efficient vector database
- *PyPDF2* – extract text from PDFs
- *PyTorch* – backend for running models

## 🔹 How It Works
1. User uploads a PDF.
2. Text is extracted and split into chunks.
3. Chunks are converted into vector embeddings.
4. A question is asked through the Streamlit UI.
5. FAISS retrieves the most relevant chunks.
6. The Transformer model generates an answer.


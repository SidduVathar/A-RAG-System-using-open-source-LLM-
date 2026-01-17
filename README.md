A RAG System using Open-Source LLM

A PDF-based AI Chat Assistant built with Streamlit and open-source LLMs
Leverages Retrieval-Augmented Generation (RAG) to answer user queries from PDF documents using local Llama2 models (7B & 14B).

🚀 Overview

This project enables you to create an AI assistant that:

📄 Ingests multiple PDF documents
🔍 Builds a vector index (FAISS) of document content
🤖 Retrieves relevant chunks based on a user query
💬 Generates contextual, grounded responses using an open-source LLM (Llama2)
🌐 Provides a web interface via Streamlit

It’s perfect for local, privacy-focused knowledge assistants with no cloud dependency.

⭐ Features
🧠 Retrieval-Augmented Generation (RAG) pipeline
📑 PDF indexing for semantic search
🧰 Uses FAISS for efficient vector search
👨‍💻 Local inference with Llama2-7B / 14B models
💡 Simple Streamlit UI for query input and answer display
🔒 No data leaves your machine — full privacy



📝 How It Works

Document Processing
PDF files are read and split into text chunks.
Embeddings are created for each chunk.
Chunks are indexed in a vector database (FAISS).
Query Workflow
User enters a query in the Streamlit UI.
Similar chunks are retrieved from the FAISS index.
Rerieved chunks + user query are passed to the LLM.
LLM generates context-aware responses.

Tips & Notes

⚡ Performance depends on your hardware (GPU recommended).
🛡 For large corpora, consider FAISS parameters tuning.
📊 Embedding quality & chunk size affect retrieval relevance.

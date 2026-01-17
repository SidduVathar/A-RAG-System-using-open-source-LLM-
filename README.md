A RAG System using Open-Source LLM

A PDF-based AI Chat Assistant built with Streamlit and open-source LLMs
Leverages Retrieval-Augmented Generation (RAG) to answer user queries from PDF documents using local Llama2 models (7B & 14B).

🚀 Overview

This project enables you to create an AI assistant that:

1.Ingests multiple PDF documents
2. Builds a vector index (FAISS) of document content
3. Retrieves relevant chunks based on a user query
4. Generates contextual, grounded responses using an open-source LLM (Llama2)
5. Provides a web interface via Streamlit

It’s perfect for local, privacy-focused knowledge assistants with no cloud dependency.

1.Features
2. Retrieval-Augmented Generation (RAG) pipeline
3. PDF indexing for semantic search
4. Uses FAISS for efficient vector search
5. Local inference with Llama2-7B / 14B models
6. Simple Streamlit UI for query input and answer display
7. No data leaves your machine — full privacy



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

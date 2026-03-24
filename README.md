

```markdown
# GenAI Part 2: RAG Implementation

A comprehensive implementation of Retrieval-Augmented Generation (RAG) system that enhances Large Language Models with external knowledge retrieval capabilities.

## Overview

This repository contains a complete implementation of a RAG (Retrieval-Augmented Generation) system that combines the power of Large Language Models with external knowledge retrieval. The system processes documents, creates vector embeddings, and enables context-aware question answering using state-of-the-art NLP techniques.

## Features

- 📚 **Document Processing**: Support for multiple document formats (PDF, TXT, DOCX)
- 🧠 **Vector Embeddings**: Advanced embedding generation using transformer models
- 🔍 **Semantic Search**: Efficient similarity search using FAISS vector database
- 💬 **Chat Interface**: Interactive web-based chat interface for querying
- 🐳 **Docker Support**: Containerized deployment for easy setup
- ⚡ **Real-time Processing**: Fast document indexing and retrieval

## Architecture

The RAG system follows a modular architecture with the following components:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documents     │───▶│  Text Processing │───▶│  Vector Store   │
│   (PDF/TXT)     │    │   & Chunking     │    │   (FAISS)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                                              │
        ▼                                              ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Embedding     │◀───│  Similarity      │◀───│  User Query     │
│   Generation    │    │   Search         │    │   Interface     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                                              │
        ▼                                              ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   LLM Model     │◀───│  Context         │◀───│  RAG Chain      │
│   (Phi-2)       │    │   Retrieval      │    │   Pipeline      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Prerequisites

- Python 3.8 or higher
- 4GB+ RAM recommended
- CUDA-compatible GPU (optional, for faster processing)

## Installation

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/rahulsvt-1907/RagwithPDF


# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

#To run project
streamlit app.py


## Performance Optimization

### For Large Document Collections:
- Use GPU acceleration for embedding generation
- Implement document chunking strategies
- Configure FAISS index parameters
- Use approximate search for faster retrieval

### For Real-time Applications:
- Pre-compute document embeddings
- Use efficient similarity search algorithms
- Implement caching mechanisms
- Optimize LLM inference parameters

## Troubleshooting


## Pipeline Guide & Interview Q&A

For a detailed walkthrough of every stage of the RAG pipeline, the exact functions and classes used at each stage, and a comprehensive set of interview questions and answers, see **[PIPELINE_GUIDE.md](PIPELINE_GUIDE.md)**.

## Acknowledgments

- Hugging Face for transformer models
- LangChain framework for RAG implementation
- Streamlit for the web interface



# GenAI Part 2: RAG Implementation

A comprehensive implementation of Retrieval-Augmented Generation (RAG) system that enhances Large Language Models with external knowledge retrieval capabilities.

## Overview

This repository contains a complete implementation of a RAG (Retrieval-Augmented Generation) system that combines the power of Large Language Models with external knowledge retrieval. The system processes PDF documents, creates vector embeddings, stores them in a ChromaDB vector database, and enables context-aware question answering using the Mistral LLM.

## Features

- 📚 **Document Processing**: PDF document loading and intelligent chunking
- 🧠 **Vector Embeddings**: Sentence-transformer embedding generation (`all-MiniLM-L6-v2`)
- 🔍 **Semantic Search**: Similarity search using ChromaDB vector database
- 💬 **Chat Interface**: Interactive Streamlit web UI for querying
- ⚡ **Real-time Processing**: Fast document indexing and retrieval
- 🔄 **Multiple Retrieval Strategies**: Similarity, MMR, and Multi-Query retrievers

## Architecture

The RAG system follows this pipeline:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Document  │───▶│  Text Chunking   │───▶│  Embedding      │
│  (PyPDFLoader)  │    │ (RecursiveChar   │    │  Generation     │
│                 │    │  TextSplitter)   │    │ (HuggingFace)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Final Answer   │◀───│  LLM Generation  │◀───│  Vector Store   │
│  (response      │    │  (ChatMistralAI) │    │  (ChromaDB)     │
│   .content)     │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                 ▲                      │
                                 │                      ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  User Query     │───▶│ Prompt Template  │◀───│   Retriever     │
│                 │    │(ChatPromptTemplate│    │ (similarity k=6)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

---

## Detailed Pipeline Explanation

### Phase 1 — Document Ingestion (`create_database.py`)

#### Step 1: Load PDF — `PyPDFLoader`
```python
from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("document_loaders/deeplearning.pdf")
docs = data.load()
```
- **What it does**: Reads a PDF file page-by-page and returns a list of `Document` objects.
- **Each `Document`** contains `page_content` (raw text) and `metadata` (e.g., page number, source path).
- **Why**: Raw PDFs cannot be directly understood by an LLM — they must be converted to plain text first.

#### Step 2: Split into Chunks — `RecursiveCharacterTextSplitter`
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)
```
- **What it does**: Splits large documents into smaller overlapping text chunks.
- **`chunk_size=1000`**: Each chunk is at most 1000 characters.
- **`chunk_overlap=200`**: Consecutive chunks share 200 characters to preserve context across boundaries.
- **Why**: LLMs have a limited context window. Chunking ensures each piece fits while overlap avoids losing information at boundaries.

#### Step 3: Create Embeddings — `HuggingFaceEmbeddings`
```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
```
- **What it does**: Converts text chunks into dense numerical vectors (embeddings) using the `all-MiniLM-L6-v2` sentence-transformer model.
- **`normalize_embeddings=True`**: Makes all vectors unit-length so cosine similarity equals dot-product similarity.
- **Why**: Vectors allow mathematical similarity comparisons between a user's question and the stored document chunks.

#### Step 4: Store in ChromaDB — `Chroma.from_documents`
```python
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)
```
- **What it does**: Embeds every chunk and saves the resulting vectors (along with the original text) into a persistent ChromaDB vector database on disk.
- **`persist_directory`**: Specifies the folder where the database is saved so it survives between runs.
- **Why**: Persisting the vector store avoids re-processing the PDF on every query.

---

### Phase 2 — Query & Answer (`main.py` / `app.py`)

#### Step 5: Load Vector Store & Create Retriever
```python
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)
```
- **`Chroma(...)`**: Loads the existing persisted vector database from disk.
- **`as_retriever`**: Wraps the vector store so it can be queried using LangChain's retriever interface.
- **`search_type="similarity"`**: Uses cosine similarity to find the most relevant chunks.
- **`k=6`**: Returns the top 6 most similar chunks for every query.

#### Step 6: Define the Prompt Template — `ChatPromptTemplate`
```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant.
Use the provided context as the primary source to answer the question.
If the context does not contain enough information, say:
"I could not find the answer in the document." """),
    ("human", """Context:\n{context}\n\nQuestion:\n{question}""")
])
```
- **What it does**: Defines a structured multi-turn message template with `{context}` and `{question}` as dynamic placeholders.
- **System message**: Gives the LLM a persona and strict instruction to stay within the retrieved context.
- **Human message**: Injects the retrieved context and the user's question at runtime.
- **Why**: A well-crafted prompt prevents hallucination and ensures the model stays grounded in the document.

#### Step 7: Initialize the LLM — `ChatMistralAI`
```python
from langchain_mistralai import ChatMistralAI

llm = ChatMistralAI(model="mistral-small-2506")
```
- **What it does**: Creates a LangChain-compatible client for the Mistral API.
- **Why Mistral**: It is a strong open-weight model that performs well on instruction-following tasks, balancing speed and quality.

#### Step 8: Retrieve → Build Context → Invoke LLM
```python
# 1. Retrieve relevant chunks
docs = retriever.invoke(query)

# 2. Join chunks into a single context string
context = "\n\n".join([doc.page_content for doc in docs])

# 3. Fill the prompt template
final_prompt = prompt.invoke({"context": context, "question": query})

# 4. Get the LLM response
response = llm.invoke(final_prompt)

print(response.content)
```
- **`retriever.invoke(query)`**: Embeds the user query and retrieves the top-k matching document chunks.
- **Context assembly**: Concatenates chunk texts with double newlines as separators.
- **`prompt.invoke(...)`**: Substitutes the placeholders with real values and returns a `ChatPromptValue`.
- **`llm.invoke(final_prompt)`**: Sends the filled prompt to the Mistral API and returns an `AIMessage`.
- **`response.content`**: Extracts the plain text answer from the `AIMessage` object.

---

## Retrieval Strategies (in `retrievers/`)

| Strategy | File | Key Function | Best For |
|---|---|---|---|
| Similarity Search | `mmr.py` | `search_type="similarity"` | Standard closest-match retrieval |
| MMR (Maximal Marginal Relevance) | `mmr.py` | `search_type="mmr"` | Diverse, non-redundant results |
| Multi-Query | `multiquery.py` | `MultiQueryRetriever.from_llm` | Reformulating query to capture more coverage |
| ArXiv | `arixv.py` | `ArxivRetriever` | Fetching academic papers from the web |

### MMR Retriever
```python
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3}
)
```
MMR balances relevance and diversity — it picks chunks that are similar to the query **and** different from each other, avoiding repetitive context.

### Multi-Query Retriever
```python
from langchain.retrievers.multi_query import MultiQueryRetriever

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)
```
Uses the LLM itself to generate multiple rephrased versions of the user's query and merges the retrieval results, improving recall for ambiguous questions.

---

## Prerequisites

- Python 3.8 or higher
- 4 GB+ RAM recommended
- A [Mistral API key](https://console.mistral.ai/) in a `.env` file: `MISTRAL_API_KEY=your_key`

## Installation

```bash
# Clone the repository
git clone https://github.com/rahulsvt-1907/RagwithPDF

# Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

---

## Performance Optimization

### For Large Document Collections:
- Use GPU acceleration for embedding generation (`model_kwargs={"device": "cuda"}`)
- Tune `chunk_size` and `chunk_overlap` for your document type
- Use MMR retrieval to reduce redundancy in the context

### For Real-time Applications:
- Pre-compute and persist document embeddings
- Use caching for repeated queries
- Increase `k` in the retriever for better recall at the cost of more tokens

---

## Pipeline Guide & Interview Q&A

For a detailed walkthrough of every stage of the RAG pipeline, the exact functions and classes used at each stage, and a comprehensive set of interview questions and answers, see **[PIPELINE_GUIDE.md](PIPELINE_GUIDE.md)**.

## Acknowledgments

- Hugging Face for sentence-transformer embedding models
- LangChain framework for RAG orchestration
- ChromaDB for the persistent vector store
- Mistral AI for the language model
- Streamlit for the web interface

---

## Interview Questions & Answers

### 1. What is RAG and why is it used?

**Answer:**  
RAG stands for **Retrieval-Augmented Generation**. It is a technique that combines a retrieval system (vector database) with a generative LLM. Instead of relying solely on the LLM's parametric (trained) knowledge — which may be outdated or hallucinated — RAG fetches relevant passages from an external document at query time and injects them into the prompt as context.

**Why use it?**  
- Reduces hallucination by grounding answers in real documents.
- Allows the model to answer questions about private or recent data without fine-tuning.
- Cost-effective compared to retraining or fine-tuning the entire model.

---

### 2. Explain the end-to-end pipeline of this RAG system.

**Answer:**  
The pipeline has two phases:

**Ingestion phase:**
1. `PyPDFLoader` loads the PDF and extracts text page-by-page.
2. `RecursiveCharacterTextSplitter` breaks the text into overlapping 1000-character chunks.
3. `HuggingFaceEmbeddings` converts each chunk into a 384-dimensional vector using `all-MiniLM-L6-v2`.
4. `Chroma.from_documents` stores the vectors and text in a persistent ChromaDB database.

**Query phase:**
1. The user's query is embedded using the same model.
2. `retriever.invoke(query)` performs cosine similarity search and returns the top 6 matching chunks.
3. Chunks are joined into a context string and inserted into a `ChatPromptTemplate`.
4. `ChatMistralAI.invoke(prompt)` generates the final answer grounded in the retrieved context.

---

### 3. What is `RecursiveCharacterTextSplitter` and why is it preferred over a simple character splitter?

**Answer:**  
`RecursiveCharacterTextSplitter` attempts to split text by a priority list of separators: `["\n\n", "\n", " ", ""]`. It first tries to split on double newlines (paragraph boundaries), then single newlines, then spaces, and only as a last resort splits mid-word.

This is preferred because it tries to keep semantically related text (paragraphs, sentences) together within a chunk, rather than cutting arbitrarily at a fixed character count. The result is more coherent chunks that embed and retrieve better.

---

### 4. What is an embedding and why do we normalize them?

**Answer:**  
An embedding is a dense vector representation of text produced by a neural network. Semantically similar texts produce vectors that are close to each other in the vector space.

Normalizing embeddings (setting `normalize_embeddings=True`) makes every vector have a magnitude of 1. When vectors are normalized, **cosine similarity** (which measures the angle between vectors) becomes equivalent to a simple **dot product**, which is faster to compute. It also ensures consistent similarity scores regardless of the length of the input text.

---

### 5. Why is `chunk_overlap=200` set in the text splitter?

**Answer:**  
When a document is split into chunks, information that sits near the boundary of two consecutive chunks would be split across them. With `chunk_overlap=200`, the last 200 characters of one chunk are repeated at the start of the next. This ensures that no piece of context is entirely "lost" at the seam between chunks, improving the quality of retrieval when the answer lies near a chunk boundary.

---

### 6. What is ChromaDB and how does it work?

**Answer:**  
ChromaDB is an open-source, embeddable vector database. It stores document embeddings along with metadata and text content. When a query embedding is provided, ChromaDB performs an Approximate Nearest Neighbor (ANN) search using algorithms like HNSW to return the most similar stored vectors efficiently.

In this project, ChromaDB is persisted to disk (`persist_directory="chroma_db"`) so the vector index does not have to be rebuilt on every run.

---

### 7. What is the difference between similarity search and MMR?

**Answer:**  
- **Similarity search**: Returns the `k` chunks most similar to the query, measured by cosine distance. It can return redundant results if several chunks say the same thing.
- **Maximal Marginal Relevance (MMR)**: Iteratively selects chunks that are both relevant to the query **and** maximally different from already-selected chunks. This produces a more diverse and informative context window, avoiding repetition.

Use MMR when the top similar chunks tend to be near-duplicates and you want broader coverage of the topic.

---

### 8. What is a `ChatPromptTemplate` and what role does it play?

**Answer:**  
`ChatPromptTemplate` is a LangChain utility for building structured, multi-turn prompt messages. It separates the **system** instruction (AI persona and rules) from the **human** message (user input), and supports dynamic `{variable}` placeholders that are filled at runtime via `prompt.invoke({...})`.

In this RAG system it plays a critical role: it ensures the LLM always receives both the retrieved context and the user question in a consistent format, and the system message instructs the model to avoid hallucinating by only using the provided context.

---

### 9. What is the `MultiQueryRetriever` and when would you use it?

**Answer:**  
`MultiQueryRetriever.from_llm(retriever, llm)` uses the LLM to automatically generate multiple semantically different phrasings of the original user query. It then retrieves documents for each rephrased query and returns the union of all results (deduplicated).

**When to use it:** When a single-phrasing query might miss relevant chunks because the document uses different terminology than the user. For example, a user asking "How do neural networks learn?" might also benefit from rephrased queries like "What is backpropagation?" or "How are weights updated in deep learning?"

---

### 10. Why do we use `retriever.invoke(query)` instead of `vectorstore.similarity_search(query)`?

**Answer:**  
Both achieve similar results, but `retriever.invoke()` follows the standard LangChain **Runnable** interface. This makes the retriever composable — it can be chained with other LangChain components using LCEL (LangChain Expression Language), supports async natively, and integrates cleanly with LangChain tracing and callbacks. `similarity_search` is a lower-level method specific to the vector store class.

---

### 11. How does `response.content` work and what type does `llm.invoke()` return?

**Answer:**  
`llm.invoke(final_prompt)` sends the formatted prompt to the Mistral API and returns an **`AIMessage`** object — a LangChain base message type. The `AIMessage` object carries metadata (model name, token usage, etc.) as well as the actual text. `.content` is the attribute that holds the plain-text string of the LLM's reply.

---

### 12. What would happen if the vector database does not contain relevant context for a query?

**Answer:**  
If the retriever returns chunks that have low similarity to the query (or returns nothing), two things happen in this implementation:

1. In `app.py`, there is an explicit check:
   ```python
   if not docs:
       st.write("I could not find the answer in the document.")
       st.stop()
   ```
2. Even if some chunks are returned but do not contain the answer, the system prompt instructs the LLM:
   > "If the context does not contain enough information, say: 'I could not find the answer in the document.'"

This two-layer safeguard prevents both empty-context crashes and LLM hallucination.

---

### 13. What is the role of `.env` and `load_dotenv()`?

**Answer:**  
Sensitive credentials like `MISTRAL_API_KEY` should never be hardcoded in source files. The `python-dotenv` library reads a `.env` file from the project root and injects the key-value pairs as environment variables. `load_dotenv()` must be called at the start of the script before any API client is initialized so the LangChain library can read `os.environ["MISTRAL_API_KEY"]` automatically.

---

### 14. How would you scale this RAG system for a large document collection (e.g., 1000 PDFs)?

**Answer:**  
- **Batch ingestion**: Process PDFs in parallel and upsert embeddings in bulk.
- **GPU embeddings**: Switch `model_kwargs={"device": "cuda"}` to generate embeddings significantly faster.
- **Chunking strategy**: Use larger `chunk_size` to reduce the total number of vectors.
- **Vector DB scaling**: Switch to a production vector database like Pinecone, Weaviate, or Qdrant with horizontal scaling.
- **Metadata filtering**: Attach metadata (e.g., document title, date) and pre-filter by metadata before similarity search to narrow the search space.
- **Caching**: Cache embeddings for frequently asked queries to avoid redundant API calls.

---

### 15. What are the limitations of this RAG pipeline?

**Answer:**  
- **Context window limit**: Even with `k=6` chunks at 1000 characters each, very long answers may still exceed the LLM's context window.
- **Chunking quality**: `RecursiveCharacterTextSplitter` is text-length based, so it can still split mid-sentence. Semantic chunking would be more accurate.
- **No re-ranking**: Retrieved chunks are not ranked by a cross-encoder for fine-grained relevance — adding a re-ranker would improve precision.
- **Single-document scope**: The current implementation does not distinguish between multiple uploaded documents when answering queries.
- **No conversation history**: Each query is independent; follow-up questions lose prior context.

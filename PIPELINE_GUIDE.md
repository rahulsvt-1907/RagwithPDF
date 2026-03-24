# RAG with PDF – Complete Pipeline Guide & Interview Q&A

This guide explains every stage of the Retrieval-Augmented Generation (RAG) pipeline used in this project, the exact functions/classes involved at each stage, and a curated set of interview questions and answers covering the entire pipeline.

---

## Table of Contents

1. [What is RAG?](#1-what-is-rag)
2. [End-to-End Pipeline Overview](#2-end-to-end-pipeline-overview)
3. [Stage-by-Stage Explanation with Functions](#3-stage-by-stage-explanation-with-functions)
   - [Stage 1 – Document Loading](#stage-1--document-loading)
   - [Stage 2 – Text Splitting / Chunking](#stage-2--text-splitting--chunking)
   - [Stage 3 – Embedding Generation](#stage-3--embedding-generation)
   - [Stage 4 – Vector Store](#stage-4--vector-store)
   - [Stage 5 – Retrieval](#stage-5--retrieval)
   - [Stage 6 – Prompt Construction](#stage-6--prompt-construction)
   - [Stage 7 – LLM Inference](#stage-7--llm-inference)
   - [Stage 8 – UI Layer (Streamlit)](#stage-8--ui-layer-streamlit)
4. [File-by-File Reference](#4-file-by-file-reference)
5. [Interview Questions & Answers](#5-interview-questions--answers)

---

## 1. What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that enhances a Large Language Model (LLM) by providing it with relevant context retrieved from an external knowledge base at inference time. Instead of relying solely on the knowledge baked into the model's weights during training, RAG:

1. Converts source documents into searchable vector embeddings and stores them in a vector database.
2. At query time, converts the user question into an embedding and performs a similarity search to find the most relevant document chunks.
3. Injects those chunks as "context" into the LLM prompt so the model can produce a grounded, accurate answer.

---

## 2. End-to-End Pipeline Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        INDEXING PHASE (offline)                          │
│                                                                          │
│  PDF / TXT / Web                                                         │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────┐   load()    ┌──────────────────────┐               │
│  │ Document Loader │ ──────────▶ │  Raw Document Pages  │               │
│  │ (PyPDFLoader /  │             │  (LangChain Document  │               │
│  │  TextLoader /   │             │   objects)           │               │
│  │  WebBaseLoader) │             └──────────────────────┘               │
│  └─────────────────┘                       │                             │
│                                            │ split_documents()           │
│                                            ▼                             │
│                              ┌──────────────────────────┐               │
│                              │  Text Splitter           │               │
│                              │  (RecursiveCharacter /   │               │
│                              │   CharacterTextSplitter) │               │
│                              └──────────────────────────┘               │
│                                            │                             │
│                                            │ chunks (List[Document])     │
│                                            ▼                             │
│                              ┌──────────────────────────┐               │
│                              │  Embedding Model         │               │
│                              │  (HuggingFaceEmbeddings  │               │
│                              │   all-MiniLM-L6-v2)      │               │
│                              └──────────────────────────┘               │
│                                            │                             │
│                                            │ dense vectors               │
│                                            ▼                             │
│                              ┌──────────────────────────┐               │
│                              │  Vector Store            │               │
│                              │  Chroma.from_documents() │               │
│                              │  (persisted to chroma_db)│               │
│                              └──────────────────────────┘               │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                        QUERYING PHASE (online)                           │
│                                                                          │
│  User Question                                                           │
│       │                                                                  │
│       ▼                                                                  │
│  ┌──────────────────┐  invoke()  ┌─────────────────────────────────┐    │
│  │    Retriever     │ ─────────▶ │  Top-k Relevant Chunks          │    │
│  │  (similarity /   │            │  (List[Document])               │    │
│  │   MMR /          │            └─────────────────────────────────┘    │
│  │   MultiQuery)    │                         │                          │
│  └──────────────────┘                         │ context string           │
│                                               ▼                          │
│                              ┌──────────────────────────┐               │
│                              │  Prompt Template         │               │
│                              │  ChatPromptTemplate      │               │
│                              │  .from_messages()        │               │
│                              └──────────────────────────┘               │
│                                               │                          │
│                                               │ formatted prompt         │
│                                               ▼                          │
│                              ┌──────────────────────────┐               │
│                              │  LLM                     │               │
│                              │  ChatMistralAI           │               │
│                              │  (mistral-small-2506)    │               │
│                              └──────────────────────────┘               │
│                                               │                          │
│                                               │ response.content         │
│                                               ▼                          │
│                              ┌──────────────────────────┐               │
│                              │  Answer displayed to     │               │
│                              │  user (CLI / Streamlit)  │               │
│                              └──────────────────────────┘               │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Stage-by-Stage Explanation with Functions

---

### Stage 1 – Document Loading

**Goal:** Read raw content from a file or URL and convert it into LangChain `Document` objects.

| Class / Function | Source File | What It Does |
|---|---|---|
| `PyPDFLoader(file_path)` | `create_database.py`, `app.py`, `document_loaders/pdf.py` | Reads a PDF page-by-page using `pypdf`. Each page becomes one `Document`. |
| `data.load()` | All loaders | Returns `List[Document]`; each `Document` has `.page_content` (text) and `.metadata` (source, page number, etc.). |
| `TextLoader(file_path)` | `document_loaders/test.py` | Loads a plain-text file as a single `Document`. |
| `WebBaseLoader(url)` | `document_loaders/page.py` | Fetches a webpage with `requests` + `BeautifulSoup` and creates a `Document` from its text. |

**Key concepts:**
- A `Document` is the fundamental data unit in LangChain. It carries both the raw text and metadata.
- `PyPDFLoader` preserves page-level metadata (page index, source path), which helps trace answers back to a specific page.

---

### Stage 2 – Text Splitting / Chunking

**Goal:** Break large documents into smaller, overlapping pieces (chunks) that fit within embedding and LLM context windows while preserving enough local context.

| Class / Function | Source File | What It Does |
|---|---|---|
| `RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)` | `create_database.py`, `app.py`, `document_loaders/pdf.py` | Splits text hierarchically by `\n\n` → `\n` → ` ` → character until each chunk is ≤ `chunk_size`. |
| `CharacterTextSplitter(separator, chunk_size, chunk_overlap)` | `document_loaders/test.py` | Simpler splitter that splits only on a single specified separator. |
| `splitter.split_documents(docs)` | All splitter files | Applies the splitter to a list of `Document` objects and returns a new (longer) list of smaller `Document` chunks. |

**Parameters used in this project:**
- `chunk_size=1000` – maximum characters per chunk.
- `chunk_overlap=200` – characters shared between consecutive chunks to avoid information loss at boundaries.

**Why overlap matters:** Without overlap, a sentence split across two chunks would lose context. A 200-character overlap ensures the tail of one chunk appears at the head of the next.

---

### Stage 3 – Embedding Generation

**Goal:** Convert text chunks into dense numerical vectors so that semantic similarity can be computed efficiently.

| Class / Function | Source File | What It Does |
|---|---|---|
| `HuggingFaceEmbeddings(model_name, model_kwargs, encode_kwargs)` | `create_database.py`, `app.py`, `main.py`, `vector store/DB.py` | Downloads and wraps a Sentence-Transformers model from Hugging Face. |
| `model_name="sentence-transformers/all-MiniLM-L6-v2"` | All embedding files | A lightweight, fast 384-dimensional model good for semantic search. |
| `model_kwargs={"device": "cpu"}` | `create_database.py`, `app.py`, `main.py` | Forces CPU inference (no GPU required). |
| `encode_kwargs={"normalize_embeddings": True}` | `create_database.py`, `app.py`, `main.py` | L2-normalizes vectors so cosine similarity = dot product (faster at search time). |

**How embeddings work:**
1. The text is tokenized into sub-word tokens.
2. The transformer encoder produces a token-level representation.
3. Mean pooling over token representations gives a single sentence-level vector.
4. The vector captures semantic meaning – similar sentences produce similar vectors.

---

### Stage 4 – Vector Store

**Goal:** Persist the embeddings alongside their source text so they can be queried efficiently at runtime.

| Class / Function | Source File | What It Does |
|---|---|---|
| `Chroma.from_documents(documents, embedding, persist_directory)` | `create_database.py`, `app.py`, `retrievers/mmr.py`, `retrievers/multiquery.py` | Embeds all chunks, builds an HNSW index, and stores everything in `chroma_db/`. |
| `vectorstore.persist()` | `app.py` | Explicitly flushes the in-memory Chroma store to disk (required for older Chroma versions). |
| `Chroma(persist_directory, embedding_function)` | `main.py`, `app.py` | Re-loads a previously persisted Chroma store from disk. |
| `vectorstore.similarity_search(query, k)` | `vector store/DB.py` | Direct similarity search returning top-k `Document` objects. |
| `vectorstore.as_retriever(search_type, search_kwargs)` | `main.py`, `app.py`, `retrievers/mmr.py` | Wraps the vector store as a LangChain `Retriever` object with configurable search strategy. |

**ChromaDB internals:**
- Uses **HNSW** (Hierarchical Navigable Small World graph) for approximate nearest-neighbour search.
- Stores embeddings in `chroma_db/<collection-uuid>/data_level0.bin` and metadata in `chroma_db/chroma.sqlite3`.

---

### Stage 5 – Retrieval

**Goal:** At query time, convert the question into an embedding and find the most relevant chunks.

This project implements three retrieval strategies:

#### 5a. Similarity Search (used in `main.py` and `app.py`)

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6},
)
docs = retriever.invoke(query)
```

- Computes cosine similarity between the query vector and every stored chunk vector.
- Returns the top `k=6` chunks.
- Fast and simple; can return redundant documents.

#### 5b. MMR – Maximal Marginal Relevance (used in `retrievers/mmr.py`)

```python
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3}
)
```

- Balances **relevance** (similarity to the query) and **diversity** (dissimilarity to already-selected chunks).
- Iteratively selects the chunk that maximises: `λ · sim(chunk, query) − (1−λ) · max sim(chunk, selected)`.
- Avoids returning near-duplicate chunks that carry the same information.

#### 5c. MultiQuery Retriever (used in `retrievers/multiquery.py`)

```python
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)
docs = multi_query_retriever.invoke(query)
```

- Uses the LLM to **rephrase the original question into multiple alternative queries**.
- Runs each rephrased query through the base retriever and deduplicates results.
- Reduces the sensitivity to exact wording of the user's question.

#### 5d. Arxiv Retriever (used in `retrievers/arixv.py`)

```python
retriever = ArxivRetriever(load_max_docs=2, load_all_available_meta=True)
docs = retriever.invoke("large language models")
```

- Retrieves academic papers directly from the **Arxiv API** without a local vector store.
- Useful for augmenting the RAG system with up-to-date research.

| Method | `retriever.invoke(query)` | Returns |
|---|---|---|
| All retriever types | Takes a string query | `List[Document]` |

---

### Stage 6 – Prompt Construction

**Goal:** Combine the retrieved context and the user question into a structured prompt the LLM can follow.

| Class / Function | Source File | What It Does |
|---|---|---|
| `ChatPromptTemplate.from_messages([...])` | `main.py`, `app.py` | Builds a chat-style prompt template from a list of `(role, text)` tuples. |
| `prompt.invoke({"context": ..., "question": ...})` | `main.py`, `app.py` | Fills in the `{context}` and `{question}` placeholders and returns a `ChatPromptValue`. |

**Prompt structure used:**
```
System: You are a helpful AI assistant.
        Use ONLY the provided context to answer the question.
        If the answer is not present, say "I could not find the answer."
```
```
Human:  Context: {context}
        Question: {question}
```

**Why a system + human message structure?**  
Chat models are fine-tuned on conversations. Separating "who you are" (system) from "what I'm asking" (human) produces better-grounded responses than a single flat prompt.

---

### Stage 7 – LLM Inference

**Goal:** Send the filled prompt to the LLM and get a natural-language answer.

| Class / Function | Source File | What It Does |
|---|---|---|
| `ChatMistralAI(model="mistral-small-2506")` | `main.py`, `app.py` | Initialises the Mistral client using `MISTRAL_API_KEY` from `.env`. |
| `llm.invoke(final_prompt)` | `main.py`, `app.py` | Sends the prompt to the Mistral API and returns an `AIMessage`. |
| `response.content` | `main.py`, `app.py` | Extracts the text string from the `AIMessage` response object. |

**Why Mistral?**  
`mistral-small-2506` is a fast, cost-effective model that handles long context windows well – ideal for RAG where the prompt includes several retrieved chunks.

---

### Stage 8 – UI Layer (Streamlit)

**Goal:** Provide an interactive browser-based interface for uploading PDFs and asking questions.

| Function / Widget | Source File | What It Does |
|---|---|---|
| `st.set_page_config(page_title=...)` | `app.py` | Sets the browser tab title. |
| `st.title(...)` / `st.write(...)` | `app.py` | Renders markdown text on the page. |
| `st.file_uploader("...", type="pdf")` | `app.py` | Shows a file-upload widget; returns a `UploadedFile` object or `None`. |
| `tempfile.NamedTemporaryFile()` | `app.py` | Saves the uploaded bytes to a temp file so `PyPDFLoader` can read it from disk. |
| `st.button("Create Vector Database")` | `app.py` | Triggers the indexing pipeline on click. |
| `st.spinner("...")` | `app.py` | Shows a loading indicator while the pipeline runs. |
| `st.success(...)` | `app.py` | Shows a green success banner. |
| `st.text_input("Enter your question")` | `app.py` | Text box that returns the typed string or an empty string. |
| `st.divider()` / `st.subheader(...)` | `app.py` | Decorative layout helpers. |
| `st.stop()` | `app.py` | Immediately halts further Streamlit script execution for the current run. |

---

## 4. File-by-File Reference

| File | Role | Key Classes / Functions |
|---|---|---|
| `create_database.py` | Offline indexing script | `PyPDFLoader`, `RecursiveCharacterTextSplitter`, `HuggingFaceEmbeddings`, `Chroma.from_documents` |
| `main.py` | CLI query loop | `Chroma`, `HuggingFaceEmbeddings`, `vectorstore.as_retriever`, `ChatMistralAI`, `ChatPromptTemplate`, `retriever.invoke`, `llm.invoke` |
| `app.py` | Streamlit web app | Everything in `main.py` + `st.*` widgets + `tempfile` |
| `document_loaders/pdf.py` | Loader demo – PDF | `PyPDFLoader`, `RecursiveCharacterTextSplitter` |
| `document_loaders/page.py` | Loader demo – Web | `WebBaseLoader` |
| `document_loaders/test.py` | Loader demo – Text | `TextLoader`, `CharacterTextSplitter` |
| `retrievers/mmr.py` | Retrieval demo – MMR | `Chroma`, `HuggingFaceEmbeddings`, `as_retriever(search_type="mmr")` |
| `retrievers/multiquery.py` | Retrieval demo – MultiQuery | `MultiQueryRetriever.from_llm`, `ChatMistralAI` |
| `retrievers/arixv.py` | Retrieval demo – Arxiv | `ArxivRetriever` |
| `vector store/DB.py` | Vector store demo | `Chroma.from_documents`, `similarity_search`, `as_retriever` |

---

## 5. Interview Questions & Answers

### Section A – RAG Fundamentals

---

**Q1. What is Retrieval-Augmented Generation (RAG) and why is it needed?**

**A:** RAG is a technique that augments an LLM's response with information retrieved from an external knowledge base at inference time. It is needed because:
- LLMs have a **knowledge cutoff** – they don't know about events after training.
- LLMs can **hallucinate** facts when answering questions outside their training data.
- Private or domain-specific documents are not part of any LLM's training data.

RAG solves these problems by retrieving the most relevant passages from your documents and injecting them as context into the prompt, grounding the answer in real source material.

---

**Q2. What are the two main phases of a RAG pipeline?**

**A:**
1. **Indexing phase (offline):** Documents are loaded → split into chunks → embedded into vectors → stored in a vector database.
2. **Querying phase (online):** The user's question is embedded → similar chunks are retrieved → chunks are injected into a prompt → the LLM generates an answer.

---

**Q3. What is a `Document` object in LangChain?**

**A:** A `Document` is LangChain's fundamental data unit. It has two fields:
- `page_content` – the raw text string.
- `metadata` – a dict of arbitrary key-value pairs (e.g., `{"source": "file.pdf", "page": 3}`).

Document loaders return `List[Document]`, and text splitters also return `List[Document]` (smaller chunks).

---

### Section B – Document Loading

---

**Q4. What does `PyPDFLoader` do and what does `data.load()` return?**

**A:** `PyPDFLoader` reads a PDF file using the `pypdf` library. It extracts text page-by-page. `data.load()` returns a `List[Document]` where each `Document` corresponds to one page of the PDF, with `metadata` containing the source path and page number.

---

**Q5. What other document loaders are used in this project and when would you choose each?**

**A:**
- **`PyPDFLoader`** – for PDF files; preserves page-level structure.
- **`TextLoader`** – for plain `.txt` files; loads the whole file as one `Document`.
- **`WebBaseLoader`** – for web pages; fetches HTML and strips tags using BeautifulSoup.

You choose based on the source format of your knowledge base.

---

### Section C – Text Splitting

---

**Q6. Why do we split documents into chunks before embedding?**

**A:** Three reasons:
1. **Context window limits:** Embedding models and LLMs have token limits. A 300-page PDF cannot be embedded as one vector or passed as one context.
2. **Retrieval granularity:** If you embed a whole document, you retrieve the whole document. Chunking lets you retrieve only the 1–2 paragraphs that are truly relevant.
3. **Semantic cohesion:** Smaller chunks are more likely to have a single, clear topic, producing more meaningful embeddings.

---

**Q7. What is `RecursiveCharacterTextSplitter` and how does it differ from `CharacterTextSplitter`?**

**A:**
- **`RecursiveCharacterTextSplitter`** tries a hierarchy of separators: `"\n\n"` → `"\n"` → `" "` → character. It keeps splitting until chunks are ≤ `chunk_size`. This produces more semantically coherent chunks because it prefers to split at paragraph or sentence boundaries.
- **`CharacterTextSplitter`** splits only on a single, fixed separator you specify. It is simpler but less adaptive to text structure.

---

**Q8. What is `chunk_overlap` and why is it important?**

**A:** `chunk_overlap` is the number of characters shared between the end of one chunk and the beginning of the next. Without overlap, a sentence or concept that straddles two chunk boundaries would be incomplete in both chunks. With overlap (200 characters in this project), each chunk includes a bit of its neighbour's content, preserving continuity and reducing the risk of missing the answer due to a boundary split.

---

### Section D – Embeddings

---

**Q9. What is a sentence embedding and how does `all-MiniLM-L6-v2` produce one?**

**A:** A sentence embedding is a fixed-length dense vector (384 dimensions for `all-MiniLM-L6-v2`) that represents the semantic meaning of a sentence or paragraph. The model:
1. Tokenizes the input text into sub-word tokens.
2. Passes tokens through 6 transformer layers (the "L6" in the name).
3. Applies mean pooling over the final hidden states to get one vector.
4. With `normalize_embeddings=True`, the vector is L2-normalised so cosine similarity equals the dot product.

Sentences with similar meaning produce vectors that are close together in this 384-dimensional space.

---

**Q10. Why is `normalize_embeddings=True` used?**

**A:** L2 normalisation makes every vector have unit length. Under this condition, **cosine similarity = dot product**. Computing dot products is significantly faster than computing full cosine similarity (which requires norms). This speeds up vector search operations, especially at scale.

---

**Q11. Why use a HuggingFace model instead of OpenAI embeddings?**

**A:**
- **Cost:** HuggingFace models run locally for free; OpenAI charges per token.
- **Privacy:** Data never leaves your machine.
- **Offline use:** No internet connection needed after the model is downloaded.
- **`all-MiniLM-L6-v2` trade-off:** It is smaller and faster than larger models while still achieving strong semantic search quality, making it ideal for a local RAG system.

---

### Section E – Vector Store (ChromaDB)

---

**Q12. What is ChromaDB and how does it store data?**

**A:** ChromaDB is an open-source, embedded vector database. It stores:
- Embeddings in binary files (`data_level0.bin`) using an **HNSW** (Hierarchical Navigable Small World) graph index for fast approximate nearest-neighbour search.
- Text content and metadata in **SQLite** (`chroma.sqlite3`).

`Chroma.from_documents()` embeds all chunks and writes them to disk under `chroma_db/`. `Chroma(persist_directory=...)` reloads them.

---

**Q13. What is the difference between `Chroma.from_documents()` and `Chroma(persist_directory=...)`?**

**A:**
- `Chroma.from_documents(documents, embedding, persist_directory)` **creates** a new collection: embeds every document and saves them. Use this during the indexing phase.
- `Chroma(persist_directory, embedding_function)` **loads** an existing collection from disk. Use this at query time to avoid re-embedding.

---

**Q14. What does `vectorstore.as_retriever()` do?**

**A:** It wraps the vector store in a LangChain `Retriever` interface, exposing a single `invoke(query: str) -> List[Document]` method. This decouples the retrieval logic from the vector store implementation, making it easy to swap retrieval strategies (`similarity`, `mmr`, etc.) without changing the rest of the pipeline.

---

### Section F – Retrieval Strategies

---

**Q15. Explain similarity search. What metric is used?**

**A:** Similarity search embeds the query into a vector and computes the **cosine similarity** (or dot product after normalisation) between the query vector and every stored chunk vector. It returns the `k` chunks with the highest similarity score. It is fast but can return redundant (near-duplicate) documents.

---

**Q16. What is MMR (Maximal Marginal Relevance) and when should you use it?**

**A:** MMR selects documents iteratively:
1. First, pick the most similar document to the query.
2. For each subsequent pick, choose the document that maximises: `λ · relevance_to_query − (1−λ) · max_similarity_to_already_selected`.

This trades off relevance against diversity. Use MMR when:
- Your document collection has many near-duplicate passages (e.g., repeated legal clauses, duplicated paragraphs).
- You want to present diverse angles on the question.

---

**Q17. What is a MultiQueryRetriever and what problem does it solve?**

**A:** A MultiQueryRetriever uses an LLM to automatically generate multiple alternative phrasings of the original question. It then runs each rephrasing through the base retriever and deduplicates the results.

**Problem it solves:** Embedding-based retrieval is sensitive to the exact wording of a query. If the document uses different terminology than the user, the relevant chunks may score low. By generating multiple rephrasings, MultiQueryRetriever increases the chance of matching the document's vocabulary.

---

**Q18. What is the `ArxivRetriever` and how is it different from the other retrievers?**

**A:** `ArxivRetriever` fetches papers directly from the **Arxiv public API** based on keyword search. Unlike other retrievers in this project, it does not use a local vector store or embeddings – it delegates search entirely to Arxiv's own search engine. It is used to augment the system with live academic literature that may not be in the local document collection.

---

### Section G – Prompt Engineering

---

**Q19. What is a `ChatPromptTemplate` and why use messages instead of a single string?**

**A:** `ChatPromptTemplate` structures the prompt as a list of role-tagged messages (system, human, AI). Chat-tuned models are trained on this conversation format and respond much better to it than to a flat string.

- **System message:** Sets the assistant's persona and ground rules ("Use only the provided context").
- **Human message:** Provides the actual context and question.

Using a template with `{context}` and `{question}` placeholders makes the prompt reusable and separates structure from data.

---

**Q20. Why does the system prompt say "use ONLY the provided context"?**

**A:** This instruction prevents the model from mixing document-grounded information with its own (potentially outdated or hallucinated) knowledge. It forces answers to be traceable to the retrieved chunks, making the system more trustworthy and auditable.

---

**Q21. What happens when `prompt.invoke({"context": ..., "question": ...})` is called?**

**A:** LangChain fills in the `{context}` and `{question}` template variables and returns a `ChatPromptValue` – an object containing the final list of messages ready to be sent to the LLM. This object is then passed directly to `llm.invoke()`.

---

### Section H – LLM & Response

---

**Q22. What is `ChatMistralAI` and what does `llm.invoke(prompt)` return?**

**A:** `ChatMistralAI` is LangChain's wrapper for the Mistral AI API. It reads the `MISTRAL_API_KEY` from the environment. `llm.invoke(prompt)` sends the filled prompt to Mistral's API and returns an `AIMessage` object. The actual text answer is in `response.content`.

---

**Q23. Why is the LLM invoked at query time rather than being pre-computed?**

**A:** The LLM's answer depends on the specific question and the specific retrieved context – both of which change with every query. The LLM cannot be pre-computed. Only the embeddings and vector index are pre-computed (offline) to make query time fast.

---

### Section I – Architecture & Design

---

**Q24. What are the advantages of storing embeddings in a vector database vs. re-embedding every query?**

**A:**
- **Speed:** Embedding all documents takes minutes; a vector store search takes milliseconds.
- **Cost:** LLM-based or API-based embedding models charge per token. Pre-computing avoids repeated charges.
- **Scalability:** The HNSW index scales to millions of vectors with sub-linear search time.

---

**Q25. What would you improve if this system needed to handle 10,000 PDFs?**

**A:**
1. **Batch embedding** with GPU acceleration to speed up indexing.
2. **Metadata filtering** (e.g., filter by document ID before doing vector search) to reduce search space.
3. **Hybrid search** – combine BM25 (keyword) with vector search (semantic) for better recall.
4. **Chunking strategy tuning** – use semantic chunking instead of fixed-size to produce more coherent chunks.
5. **Async query handling** – avoid blocking the UI while waiting for LLM responses.
6. **Caching** – cache embeddings for frequently asked questions.

---

**Q26. How would you evaluate the quality of this RAG system?**

**A:**
- **Retrieval metrics:** Precision@k, Recall@k – do the retrieved chunks contain the correct answer?
- **Answer faithfulness:** Does the generated answer contradict the retrieved context? (Use tools like RAGAS.)
- **Answer relevance:** Is the answer actually relevant to the question?
- **Context precision/recall:** Does the retrieved context contain enough (and not too much) information?

---

**Q27. What is the role of `load_dotenv()` in this project?**

**A:** `load_dotenv()` reads a `.env` file in the project root and loads its contents as environment variables. This is used to supply the `MISTRAL_API_KEY` to `ChatMistralAI` without hardcoding the secret in source code.

---

**Q28. Trace the complete flow of a single user query end-to-end.**

**A:**

1. User types a question in the Streamlit text box (or CLI input).
2. `retriever.invoke(query)` is called:
   - `query` is embedded using `HuggingFaceEmbeddings` → 384-dim vector.
   - ChromaDB performs HNSW search → returns top-6 `Document` chunks.
3. `context` string is built by joining `doc.page_content` for each chunk.
4. `prompt.invoke({"context": context, "question": query})` fills the template → `ChatPromptValue`.
5. `llm.invoke(final_prompt)` sends the prompt to the Mistral API → `AIMessage`.
6. `response.content` (the answer string) is displayed to the user.

---

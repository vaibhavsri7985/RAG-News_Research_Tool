![RAG News Research Tool](<img width="1918" height="747" alt="Screenshot 2025-09-30 033606" src="https://github.com/user-attachments/assets/3d1c05d8-a80b-4fd9-be9d-55de56f79c95" />
)
This project implements a **Retrieval-Augmented Generation (RAG)** system for answering questions from online news articles. It provides two interfaces:  

1. A **simple Python script (`simple.py`)** for command-line usage.  
2. A **Streamlit-based UI (`streamlit_app.py`)** for interactive usage.

---

## Project Overview

The News Research Tool allows users to input URLs of news articles. It processes the content, creates embeddings, stores them in a FAISS vector store, and answers queries strictly based on retrieved content. This ensures answers are grounded in the provided context and minimizes hallucinations from the language model.

Key steps in the project include:

### 1. Data Loading
- News article content is fetched from URLs using `UnstructuredURLLoader` from LangChain.
- Multiple URLs can be processed to create a unified knowledge base.

### 2. Text Splitting
- Loaded text is split into smaller chunks using `RecursiveCharacterTextSplitter`.
- Chunk size is set to 1000 characters with 200-character overlap to maintain context continuity.

### 3. Embeddings Creation
- Chunks (documents) are converted into vector embeddings using **Sentence Transformers**: `sentence-transformers/all-MiniLM-L6-v2`.
- Embeddings enable semantic search for relevant content during question answering.

### 4. Vector Store
- FAISS is used to store embeddings efficiently.
- The vector store is saved locally as `vector_store.pkl` using `pickle` for reuse across sessions.

### 5. Question-Answering Pipeline
- Queries are answered by retrieving the top relevant documents using a **similarity-based retriever**.
- Retrieved documents are formatted into a context string.
- A **HuggingFace LLM** (`meta-llama/Llama-3.1-8B-Instruct`) generates answers based on the context.
- The prompt ensures answers strictly rely on the context and returns `"I don't know"` if the context is insufficient.

### 6. RAG Chain Design
- A `RunnableParallel` chain passes both the question and the context to the model in a structured way.
- Outputs are parsed using `StrOutputParser` for clean textual responses.

### 7. Source Tracking
- Tracks source URLs for each retrieved document.
- Allows verification of the model's answers against the original articles.

---

## Features

- Retrieval-based QA to reduce hallucination.
- Supports multiple news articles as input.
- Answers are strictly context-based.
- Local persistence of embeddings via FAISS and `pickle`.
- Source tracking for transparency.
- Available as both a **script** and **interactive Streamlit app**.

---

## Technologies Used

- Python 3
- LangChain
- HuggingFace Transformers (`meta-llama/Llama-3.1-8B-Instruct`)
- Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS
- Pickle
- Streamlit (for UI version)

---

## Usage

### Streamlit UI (`streamlit_app.py`)
1. Run `streamlit run streamlit_app.py`.
2. Enter URLs of news articles in the sidebar.
3. Click **Process URLs** to load and split content, create embeddings, and store in FAISS.
4. Enter a question in the input box.
5. The answer is displayed along with sources.

---

## Notes

- The system strictly answers based on the retrieved context to prevent hallucination.
- Embeddings are stored locally to avoid repeated processing of the same URLs.
- Supports multiple documents and efficiently retrieves the most relevant content using semantic search.

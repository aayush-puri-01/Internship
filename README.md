# Recipe and RAG Pipeline API

This FastAPI application provides two pipelines: `RecipeGenerator` for generating recipes based on ingredients and `RAGPipeline` for answering queries using a Retrieval-Augmented Generation (RAG) model. Both support streaming and non-streaming responses.

## Prerequisites

- Python 3.8+
- Install dependencies: `pip install fastapi uvicorn pydantic ollama langchain langchain-community chromadb`
- Run Ollama server with `deepseek-r1:1.5b` model.
- Ensure `Attention.pdf` exists in `rag_pipeline/` for RAGPipeline.
- Ensure prompt files (`rag_pipeline/prompt_templates.txt`, `rag_pipeline/prompt_templates_oneline.txt`) exist for RAGPipeline.
- Start the server: `uvicorn main:app --host 0.0.0.0 --port 8000`

## API Endpoints

### 1. RecipeGenerator Pipeline

Generates recipes based on ingredients and a prompt.

#### Non-Streaming Request

```bash
curl -X POST "http://localhost:8000/generate-recipe" \
-H "Content-Type: application/json" \
-d '{"ingredients": ["chicken", "rice"], "prompt": "Suggest a quick dinner recipe"}'
```

#### Streaming Request

```bash
curl -X POST "http://localhost:8000/generate-recipe?stream=true" \
-H "Content-Type: application/json" \
-d '{"ingredients": ["chicken", "rice"], "prompt": "Suggest a quick dinner recipe"}'
```

### 2. RAG Pipeline

Answers queries using a RAG model based on a PDF document (Attention.pdf).

#### Non-Streaming Request

```bash
curl -X POST "http://localhost:8000/query" \
-H "Content-Type: application/json" \
-d '{"user_query": "What is attention in transformers?"}'
```

#### Streaming Request

```bash
curl -X POST "http://localhost:8000/query?stream=true" \
-H "Content-Type: application/json" \
-d '{"user_query": "What is attention in transformers?"}'
```

# RAG with LLM-based Reranker

This project implements a simple Retrieval-Augmented Generation (RAG) pipeline enhanced with an LLM-based reranking strategy to improve the relevance of retrieved documents before generating a response.

---

## How It Works

### Step-by-Step Pipeline

#### 1. **Load Document**

- The paper _"Attention is All You Need"_ is loaded from a PDF.
- Text is extracted using a PDF parser and combined into a single string.

#### 2. **Chunk and Embed**

- The document is split into overlapping text chunks using fixed-size chunking (e.g., `chunk_size=500`, `chunk_overlap=50`).
- Each chunk is converted into an embedding vector using Ollama's `nomic-embed-text` model.
- Chunks and embeddings are stored in a persistent vector store (`ChromaDB`).

#### 3. **Initial Retrieval**

- When the user submits a query, the vector database retrieves the top-`m` semantically similar chunks using cosine similarity.

#### 4. **LLM-based Reranking**

- For each of the top-`m` retrieved chunks:
  - The LLM generates a query that best represents the content of that chunk.
  - This generated query is compared to the original user query using cosine similarity of their embeddings.
- The top-`n` most relevant chunks are selected based on this reranking.

#### 5. **Answer Generation**

- A static prompt template is filled with:
  - The top-`n` reranked chunks (merged as context)
  - The original user query
- The prompt is passed to an Ollama LLM (e.g., `deepseek-r1:1.5b`).
- The LLM generates a final response based on the prompt.

---

# Cooking Assistant

A simple Python application that interacts with the Ollama API to generate recipes based on user-provided ingredients and prompts. Supports both synchronous and streaming modes for recipe generation.

## Features

- Collects ingredients from user input.
- Suggests possible dishes based on ingredients.
- Generates detailed recipes using the Ollama API.
- Supports streaming mode to display recipe output in real-time.
- Uses Pydantic for structured input/output validation.

---

# Requirements

- Python 3.8+
- `uv` package manager (`pip install uv`)
- `ollama` library
- `pydantic` library
- Running Ollama server with the `deepseek-r1:1.5b` model

## Usage

1. Clone the respository

2. Install dependencies using uv: `uv sync`

3. Run uvicorn: `uvicorn main:app --reload`

## Notes

- Ensure the Ollama server is running with the correct model.
- Streaming mode displays recipe JSON chunks as they arrive.
- Invalid inputs or API errors are handled with clear error messages.

```

```

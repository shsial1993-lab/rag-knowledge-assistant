# RAG Knowledge Assistant

A small retrieval-augmented generation foundation that separates document retrieval
from answer generation. The included retriever uses TF-IDF so it runs offline and can
later be replaced with sentence embeddings or a vector database.

## What it demonstrates

- Text normalization and TF-IDF indexing
- Cosine-similarity retrieval with configurable top-k
- Source-aware results for grounded answers
- A minimal FastAPI endpoint for integration

## Run

~~~bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m src.demo
uvicorn src.api:app --reload
~~~

Open the API documentation at http://127.0.0.1:8000/docs.
This repository intentionally returns retrieved context; connect an approved language
model separately and keep citations in the final answer.

## Structure

- src/retriever.py — TF-IDF retriever
- src/api.py — FastAPI query endpoint
- src/demo.py — offline example

## License

Apache-2.0

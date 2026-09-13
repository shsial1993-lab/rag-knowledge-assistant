from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .retriever import TfidfRetriever


DOCUMENTS = {
    'segmentation': 'Binary segmentation can be evaluated with IoU, Dice, precision, recall, and an error map.',
    'deployment': 'Edge inference can use pruning, INT8 quantization, TensorRT, and ONNX Runtime.',
    'research': 'A reproducible experiment records the data split, preprocessing, threshold, and evaluation protocol.',
}
retriever = TfidfRetriever(DOCUMENTS)
app = FastAPI(title='RAG Knowledge Assistant', version='1.0.0')


class QueryRequest(BaseModel):
    query: str = Field(min_length=2)
    top_k: int = Field(default=3, ge=1, le=10)


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.post('/query')
def query(request: QueryRequest) -> dict[str, object]:
    results = retriever.search(request.query, request.top_k)
    return {'query': request.query, 'sources': [result.__dict__ for result in results]}

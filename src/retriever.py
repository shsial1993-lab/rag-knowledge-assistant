from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class SearchResult:
    document_id: str
    score: float
    text: str


class TfidfRetriever:
    def __init__(self, documents: dict[str, str]) -> None:
        if not documents:
            raise ValueError('at least one document is required')
        self.documents = documents
        self.ids = list(documents)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.matrix = self.vectorizer.fit_transform(documents.values())

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        if not query.strip():
            raise ValueError('query must not be empty')
        if top_k < 1:
            raise ValueError('top_k must be positive')
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix)[0]
        ranked = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)
        return [
            SearchResult(self.ids[index], float(score), self.documents[self.ids[index]])
            for index, score in ranked[:top_k]
        ]

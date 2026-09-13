from src.retriever import TfidfRetriever


def test_retriever_returns_relevant_source() -> None:
    retriever = TfidfRetriever({'a': 'vision transformer patches', 'b': 'database indexing'})
    results = retriever.search('transformer patches', top_k=1)
    assert results[0].document_id == 'a'

from .api import DOCUMENTS
from .retriever import TfidfRetriever


def main() -> None:
    retriever = TfidfRetriever(DOCUMENTS)
    for result in retriever.search('How should I evaluate segmentation?', top_k=2):
        print(f'{result.document_id}: {result.score:.3f} — {result.text}')


if __name__ == '__main__':
    main()

"""Tests du découpage en passages."""
from langchain_core.documents import Document

from rag.chunking import decouper


def test_decoupage_cree_plusieurs_passages():
    texte = "Phrase un. " * 200
    docs = [Document(page_content=texte, metadata={"source": "test.pdf", "page": 1})]
    passages = decouper(docs)
    assert len(passages) > 1


def test_decoupage_conserve_les_metadonnees():
    docs = [Document(page_content="Bonjour le monde. " * 100,
                     metadata={"source": "test.pdf", "page": 1})]
    passages = decouper(docs)
    for p in passages:
        assert p.metadata["source"] == "test.pdf"
        assert p.metadata["page"] == 1


def test_decoupage_respecte_la_taille():
    from src.config import CHUNK_SIZE
    docs = [Document(page_content="a" * 5000, metadata={"source": "t.pdf", "page": 1})]
    passages = decouper(docs)
    assert all(len(p.page_content) <= CHUNK_SIZE + 100 for p in passages)


def test_decoupage_liste_vide():
    assert decouper([]) == []
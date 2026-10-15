"""Tests de la recherche vectorielle."""
import pytest

from src.retrieve import chercher


@pytest.mark.skipif(
    not __import__("pathlib").Path("models/chroma").exists(),
    reason="index vectoriel absent (normal en CI)",
)
def test_recherche_retourne_des_passages():
    passages = chercher("Qu'est-ce que le RAG ?", k=2)
    assert len(passages) > 0
    for p in passages:
        assert "texte" in p
        assert "source" in p
        assert "page" in p
        assert "distance" in p


def test_question_vide_leve_erreur():
    with pytest.raises(ValueError):
        chercher("")


def test_question_espaces_leve_erreur():
    with pytest.raises(ValueError):
        chercher("   ")
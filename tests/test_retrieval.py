"""Tests de la recherche vectorielle."""
from pathlib import Path

import pytest

from src.retrieve import SEUIL_DISTANCE, chercher

INDEX = Path("models/chroma")
pytestmark = pytest.mark.skipif(
    not INDEX.exists(),
    reason="index vectoriel absent (normal en CI)",
)


def test_recherche_retourne_des_passages():
    """Une question du corpus renvoie des passages sous le seuil."""
    passages = chercher("Qu'est-ce que la régularisation ?", k=2)
    assert len(passages) > 0
    for p in passages:
        assert "texte" in p
        assert "source" in p
        assert "page" in p
        assert "distance" in p
        assert p["distance"] <= SEUIL_DISTANCE


def test_recherche_multi_documents():
    """Le retrieval distingue plusieurs documents selon la question."""
    passages = chercher("Qu'est-ce que la régularisation ?", k=4)
    sources = {p["source"] for p in passages}
    assert len(sources) >= 1, f"aucune source trouvée : {sources}"


def test_question_hors_corpus_renvoie_vide():
    """Avec le seuil de similarité, une question hors corpus ne renvoie aucun passage."""
    passages = chercher("Quelle est la capitale du Japon ?", k=4)
    assert passages == []


def test_question_vide_leve_erreur():
    with pytest.raises(ValueError):
        chercher("")


def test_question_espaces_leve_erreur():
    with pytest.raises(ValueError):
        chercher("   ")
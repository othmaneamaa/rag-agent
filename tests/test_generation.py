"""Tests de la génération sourcée (avec LLM factice)."""
import pytest

from src.generate import formater_extraits


def test_formater_extraits_structure():
    passages = [
        {"texte": "Contenu 1", "source": "a.pdf", "page": 1, "distance": 0.1},
        {"texte": "Contenu 2", "source": "b.pdf", "page": 2, "distance": 0.2},
    ]
    resultat = formater_extraits(passages)
    assert "a.pdf" in resultat
    assert "b.pdf" in resultat
    assert "Contenu 1" in resultat
    assert "[Extrait 1]" in resultat
    assert "[Extrait 2]" in resultat


def test_repondre_question_vide():
    from src.generate import repondre
    with pytest.raises(ValueError):
        repondre("")
"""Tests des helpers de l'interface."""
import pytest

from ui.helpers import poser_question, formater_citation


def test_question_vide_leve_erreur():
    with pytest.raises(ValueError):
        poser_question("", lambda q: {})


def test_agent_factice_renvoie_sources():
    def faux_agent(question):
        return {
            "question": question,
            "reponse": "Test",
            "sources": [{"source": "a.pdf", "page": 1}],
            "passages": [],
        }
    resultat = poser_question("Bonjour", faux_agent)
    assert resultat["reponse"] == "Test"
    assert len(resultat["sources_uniques"]) == 1


def test_deduplication_sources():
    def faux_agent(question):
        return {
            "question": question,
            "reponse": "Test",
            "sources": [
                {"source": "a.pdf", "page": 1},
                {"source": "a.pdf", "page": 1},  # doublon
                {"source": "b.pdf", "page": 2},
            ],
            "passages": [],
        }
    resultat = poser_question("Test", faux_agent)
    assert len(resultat["sources_uniques"]) == 2


def test_formater_citation():
    assert formater_citation({"source": "a.pdf", "page": 3}) == "a.pdf — page 3"
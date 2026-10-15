"""Fixtures partagées par tous les tests."""
from pathlib import Path
import sys

import pytest

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))


@pytest.fixture
def faux_llm(monkeypatch):
    """Remplace rag.llm.generer par une fonction factice."""
    def fake_generer(prompt, model=None, temperature=0.1):
        return "Réponse factice pour les tests [cours_ml.pdf, page 1]"

    from rag import llm
    monkeypatch.setattr(llm, "generer", fake_generer)
    return fake_generer
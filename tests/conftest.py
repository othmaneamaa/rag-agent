"""Fixtures partagées : un encodeur léger, un petit index, un faux modèle de langage, l'API.

L'encodeur de hachage remplace le vrai modèle d'embedding : il compte les mots, sans torch ni
téléchargement, et suffit pour vérifier la logique (seuil, refus, citations, contrat de l'API).
Le vrai modèle est testé à part, dans tests/test_modele.py.
"""
import re
import zlib
from collections.abc import Sequence

import numpy as np
import pytest

from rag_agent.corpus import Passage, identifiant

DIMENSION = 256


class EncodeurHachage:
    nom = "hachage-test"

    def _encoder(self, textes: Sequence[str]) -> np.ndarray:
        vecteurs = np.zeros((len(textes), DIMENSION), dtype=np.float32)
        for i, texte in enumerate(textes):
            for mot in re.findall(r"\w{3,}", texte.lower()):
                vecteurs[i, zlib.crc32(mot.encode()) % DIMENSION] += 1
        normes = np.linalg.norm(vecteurs, axis=1, keepdims=True)
        return vecteurs / np.where(normes == 0, 1, normes)

    passages = _encoder
    questions = _encoder


TEXTES = [
    ("Antoine Lavoisier", "Antoine Lavoisier, chimiste français, est né le 26 août 1743 à Paris."),
    ("Moscou", "Moscou est la capitale de la Russie, traversée par la Moskova."),
    ("Tortue géante des Galápagos", "La tortue géante des Galápagos peut vivre plus de cent ans."),
    ("Berlin", "Berlin est la capitale de l'Allemagne depuis la réunification de 1990."),
]
PASSAGES = [Passage(identifiant(texte), titre, texte) for titre, texte in TEXTES]


@pytest.fixture
def encodeur():
    return EncodeurHachage()


@pytest.fixture
def index(tmp_path, encodeur):
    from rag_agent.index import construire

    return construire(PASSAGES, tmp_path / "chroma", encodeur)


@pytest.fixture
def faux_llm():
    """Répond en citant le premier extrait du prompt ; garde la trace des prompts reçus."""
    appels: list[str] = []

    def generer(prompt: str) -> str:
        appels.append(prompt)
        cite = re.search(r"^\[(.+?), (P[0-9a-f]{8})\]$", prompt, re.M)
        assert cite, "le prompt doit contenir des extraits au format [titre, identifiant]"
        return f"Réponse tirée de l'extrait [{cite.group(1)}, {cite.group(2)}]."

    generer.appels = appels  # type: ignore[attr-defined]
    return generer


@pytest.fixture
def client_api(monkeypatch, index, faux_llm):
    """L'API, branchée sur le petit index et le faux modèle de langage ; seuil large pour le hachage."""
    from fastapi.testclient import TestClient

    import api.main
    from api.agent import Agent
    from rag_agent import config

    monkeypatch.setattr(config, "SEUIL_DISTANCE", 0.9)
    monkeypatch.setattr(api.main, "charger_agent", lambda: Agent(index, faux_llm))
    with TestClient(api.main.app) as client:  # le « with » exécute le lifespan
        yield client

"""Raccord entre l'API et le paquet rag_agent.

L'API ne construit pas l'index et ne connaît ni ChromaDB ni Groq : elle ouvre l'index une fois
au démarrage, et passe chaque question à rag_agent.generation.repondre().
"""
from __future__ import annotations

from collections.abc import Callable

from rag_agent import config, llm
from rag_agent.generation import Reponse, repondre
from rag_agent.index import Index, Resultat, ouvrir


class Agent:
    """L'index ouvert, et la fonction qui rédige (Groq par défaut, une fausse dans les tests)."""

    def __init__(self, index: Index, generer: Callable[[str], str] = llm.generer) -> None:
        self.index = index
        self.generer = generer

    @property
    def nom_llm(self) -> str:
        if self.generer is llm.generer and not llm.configure():
            return "non configuré"
        return config.MODELE_LLM if self.generer is llm.generer else getattr(self.generer, "__name__", "?")

    def chercher(self, question: str, k: int) -> list[Resultat]:
        return self.index.chercher(question, k=k, seuil=config.SEUIL_DISTANCE)

    def repondre(self, question: str, k: int) -> Reponse:
        return repondre(question, self.index, self.generer, k=k, seuil=config.SEUIL_DISTANCE)


def charger_agent() -> Agent:
    """Échoue tout de suite si l'index manque : l'API ne démarre pas sans corpus."""
    return Agent(ouvrir())

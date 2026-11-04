"""Répondre à une question : chercher les passages, puis faire rédiger une réponse qui les cite.

Deux garde-fous contre l'invention, comme l'abstention INDETERMINE de sentiment-app :
1. aucun passage sous le seuil de distance : refus immédiat, sans appeler le modèle de langage ;
2. des passages, mais pas la réponse : le modèle doit répondre exactement HORS_CORPUS, et l'agent
   transforme ce mot en refus. Le refus est donc décidé par le code, pas laissé au style du modèle.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypedDict

from rag_agent import config
from rag_agent.index import Index, Resultat

Statut = Literal["REPONSE", "REFUS"]
MARQUEUR_REFUS = "HORS_CORPUS"
MESSAGE_REFUS = "Je ne trouve pas la réponse dans le corpus."

PROMPT = """Tu es un assistant documentaire. Réponds à la question en français, en te fondant
UNIQUEMENT sur les extraits ci-dessous. Après chaque affirmation, cite l'extrait utilisé au format
[titre, identifiant], par exemple [{exemple}].
Si les extraits ne contiennent pas la réponse, réponds exactement : {marqueur}

Extraits :
{extraits}

Question : {question}

Réponse :"""


class Source(TypedDict):
    passage_id: str
    titre: str
    distance: float


class Reponse(TypedDict):
    statut: Statut
    reponse: str
    sources: list[Source]
    passages: list[Resultat]


def formater_extraits(passages: list[Resultat]) -> str:
    return "\n\n".join(f"[{p['titre']}, {p['passage_id']}]\n{p['texte']}" for p in passages)


def construire_prompt(question: str, passages: list[Resultat]) -> str:
    exemple = f"{passages[0]['titre']}, {passages[0]['passage_id']}" if passages else "titre, P0000"
    return PROMPT.format(
        exemple=exemple, marqueur=MARQUEUR_REFUS, extraits=formater_extraits(passages), question=question
    )


def _refus(passages: list[Resultat]) -> Reponse:
    return {"statut": "REFUS", "reponse": MESSAGE_REFUS, "sources": [], "passages": passages}


def repondre(
    question: str,
    index: Index,
    generer: Callable[[str], str],
    k: int | None = None,
    seuil: float | None = None,
) -> Reponse:
    """La chaîne complète : recherche, refus éventuel, rédaction citée.

    k et seuil valent par défaut les réglages de config, lus à l'appel (pas à l'import).
    """
    k = config.TOP_K if k is None else k
    seuil = config.SEUIL_DISTANCE if seuil is None else seuil
    passages = index.chercher(question, k=k, seuil=seuil)
    if not passages:
        return _refus([])
    texte = generer(construire_prompt(question, passages)).strip()
    if not texte or texte.strip(" .").upper() == MARQUEUR_REFUS:
        return _refus(passages)
    cites = [p for p in passages if p["passage_id"] in texte] or passages
    sources: list[Source] = [
        {"passage_id": p["passage_id"], "titre": p["titre"], "distance": p["distance"]} for p in cites
    ]
    return {"statut": "REPONSE", "reponse": texte, "sources": sources, "passages": passages}

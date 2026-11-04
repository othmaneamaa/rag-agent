"""Interroge l'index avec toutes les questions d'un corpus et les questions hors corpus.

Partagé par evaluate.py et courbe_seuil.py : on cherche une fois, sans seuil, et on applique
ensuite les seuils sur les distances obtenues.
"""
from dataclasses import dataclass

import pandas as pd

from communs import HORS_CORPUS, chemin_corpus
from rag_agent import corpus, evaluation
from rag_agent.index import Index, ouvrir

PROFONDEUR = 10  # pour le MRR, on regarde les 10 premiers passages


@dataclass
class Mesures:
    source: str
    index: Index
    n_passages: int
    n_questions: int
    rangs: list[int]
    distances: list[list[float]]
    distances_hors: list[list[float]]
    n_hors: int


def mesurer(nom: str) -> Mesures:
    df = corpus.lire(chemin_corpus(nom))
    questions = corpus.questions(df)
    hors = pd.read_csv(HORS_CORPUS)["question"].tolist()
    index = ouvrir()
    if index.taille != df["passage_id"].nunique():
        raise SystemExit(
            f"l'index contient {index.taille} passages, le corpus {nom} {df['passage_id'].nunique()} : "
            f"relancez python src/ingest.py --corpus {nom}"
        )
    trouves = index.chercher_lot(questions["question"].tolist(), k=PROFONDEUR, seuil=None)
    trouves_hors = index.chercher_lot(hors, k=PROFONDEUR, seuil=None)
    rangs = [
        evaluation.rang(attendu, [p["passage_id"] for p in liste])
        for attendu, liste in zip(questions["passage_id"], trouves, strict=True)
    ]
    return Mesures(
        source=nom,
        index=index,
        n_passages=index.taille,
        n_questions=len(questions),
        rangs=rangs,
        distances=[[p["distance"] for p in liste] for liste in trouves],
        distances_hors=[[p["distance"] for p in liste] for liste in trouves_hors],
        n_hors=len(hors),
    )

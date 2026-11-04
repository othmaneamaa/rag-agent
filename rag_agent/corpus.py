"""Le corpus : les paragraphes de Wikipédia du jeu PIAF (Kaggle), et leurs questions.

Le fichier de Kaggle (train.csv) a une ligne par question : id, title, context, question, answers.
Un même paragraphe (context) porte plusieurs questions. On en tire :
- les passages à indexer, un par paragraphe distinct, avec un identifiant stable ;
- les questions, chacune reliée au passage dont elle est tirée, pour l'évaluation.
"""
from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

COLONNES = ["id", "title", "context", "question", "answers"]


@dataclass(frozen=True)
class Passage:
    """Un paragraphe du corpus, tel qu'il est indexé et cité."""

    id: str
    titre: str
    texte: str


def nettoyer_titre(titre: str) -> str:
    """Retire le balisage HTML que PIAF laisse dans certains titres (<i>, <span>, <abbr>)."""
    sans_balises = re.sub(r"<[^>]+>", "", str(titre))
    return re.sub(r"\s+", " ", html.unescape(sans_balises)).strip()


def identifiant(texte: str) -> str:
    """Identifiant stable d'un paragraphe : le même texte donne toujours le même identifiant."""
    return "P" + hashlib.sha1(texte.encode("utf-8")).hexdigest()[:8]


def lire(chemin: Path) -> pd.DataFrame:
    """Lit un fichier au format PIAF de Kaggle et vérifie ses colonnes."""
    df = pd.read_csv(chemin)
    manquantes = [c for c in COLONNES if c not in df.columns]
    if manquantes:
        raise ValueError(f"{chemin} : colonnes absentes {manquantes} ; attendu le train.csv de PIAF")
    df = df.dropna(subset=["context", "question"]).copy()
    df["titre"] = df["title"].map(nettoyer_titre)
    df["passage_id"] = df["context"].map(identifiant)
    return df


def passages(df: pd.DataFrame) -> list[Passage]:
    """Un passage par paragraphe distinct, dans l'ordre du fichier."""
    uniques = df.drop_duplicates("passage_id")
    return [Passage(r.passage_id, r.titre, r.context) for r in uniques.itertuples()]


def questions(df: pd.DataFrame) -> pd.DataFrame:
    """Les questions et l'identifiant du passage qui contient la réponse."""
    return df[["id", "question", "passage_id", "titre"]].reset_index(drop=True)

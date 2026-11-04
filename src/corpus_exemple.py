#!/usr/bin/env python
"""Extrait de PIAF un petit corpus d'exemple, versionné : data/corpus_exemple.csv.

Usage :
  python src/corpus_exemple.py

Il sert aux tests et à la CI, qui tournent ainsi sans télécharger Kaggle : 10 sujets, leurs
paragraphes et leurs questions, au même format que le train.csv de Kaggle. Le jeu est sous
licence CC0, on peut donc en versionner une partie.
"""
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
SOURCE = RACINE / "data" / "piaf" / "train.csv"
CIBLE = RACINE / "data" / "corpus_exemple.csv"
SUJETS = [
    "Algérie", "Histoire de l'Acadie", "Moscou", "Surréalisme", "Tortue géante des Galápagos",
    "Bronzage", "Calendrier hégirien", "Antoine Lavoisier", "Jeux olympiques d'été de 2016", "Berlin",
]


def main() -> int:
    if not SOURCE.exists():
        print("data/piaf/train.csv absent : lancez d'abord python src/telecharger.py")
        return 1
    df = pd.read_csv(SOURCE)
    extrait = df[df["title"].isin(SUJETS)]
    manquants = sorted(set(SUJETS) - set(extrait["title"]))
    if manquants:
        print(f"Sujets introuvables : {manquants}")
        return 1
    extrait.to_csv(CIBLE, index=False)
    print(
        f"{CIBLE.relative_to(RACINE)} : {len(SUJETS)} sujets, "
        f"{extrait['context'].nunique()} paragraphes, {len(extrait)} questions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Télécharge le jeu PIAF depuis Kaggle dans data/piaf/train.csv (hors de Git).

Usage :
  python src/telecharger.py

Jeu : « Piaf (French Questions & Answers) », publié sur Kaggle par The Devastator, licence CC0 :
https://www.kaggle.com/datasets/thedevastator/unlock-the-power-of-reading-comprehension-with-p
3 835 questions écrites sur 761 paragraphes de Wikipédia en français (projet PIAF d'Etalab).
Le téléchargement d'un jeu public ne demande pas de compte Kaggle.
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
CIBLE = RACINE / "data" / "piaf" / "train.csv"
JEU = "thedevastator/unlock-the-power-of-reading-comprehension-with-p"
URL = f"https://www.kaggle.com/api/v1/datasets/download/{JEU}"
PAGE = f"https://www.kaggle.com/datasets/{JEU}"


def main() -> int:
    if CIBLE.exists():
        print(f"{CIBLE.relative_to(RACINE)} déjà présent : rien à télécharger.")
        return 0
    print(f"Téléchargement de {PAGE} ...")
    try:
        with urllib.request.urlopen(URL, timeout=60) as reponse:
            archive = zipfile.ZipFile(io.BytesIO(reponse.read()))
    except OSError as erreur:
        print(
            f"Échec du téléchargement ({erreur}).\nTéléchargez l'archive à la main depuis {PAGE} "
            f"(bouton Download), puis placez train.csv dans {CIBLE.parent.relative_to(RACINE)}/.",
            file=sys.stderr,
        )
        return 1
    CIBLE.parent.mkdir(parents=True, exist_ok=True)
    CIBLE.write_bytes(archive.read("train.csv"))
    print(f"{CIBLE.relative_to(RACINE)} : {CIBLE.stat().st_size // 1024} Ko")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

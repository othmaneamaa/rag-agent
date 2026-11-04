"""Ce que partagent les scripts de src/ : où sont les corpus, et lequel choisir."""
import argparse
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))  # les scripts se lancent aussi sans « pip install -e . »

CORPUS = {
    "piaf": RACINE / "data" / "piaf" / "train.csv",
    "exemple": RACINE / "data" / "corpus_exemple.csv",
}
HORS_CORPUS = RACINE / "data" / "questions_hors_corpus.csv"
MODELS = RACINE / "models"


def argument_corpus(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--corpus", choices=sorted(CORPUS), default="piaf",
        help="piaf : le jeu Kaggle complet (défaut) ; exemple : l'extrait versionné, pour la CI",
    )


def chemin_corpus(nom: str) -> Path:
    chemin = CORPUS[nom]
    if not chemin.exists():
        remede = "python src/telecharger.py" if nom == "piaf" else "git checkout data/corpus_exemple.csv"
        raise SystemExit(f"{chemin.relative_to(RACINE)} absent : lancez {remede}")
    return chemin

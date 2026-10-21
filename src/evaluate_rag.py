"""Évaluation quantitative du RAG : Context Precision, Refus hors corpus."""
import csv
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))

from src.retrieve import SEUIL_DISTANCE, chercher


def charger_questions():
    """Charge data/questions_reference.csv."""
    chemin = RACINE / "data" / "questions_reference.csv"
    with chemin.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def evaluer():
    questions = charger_questions()
    total = len(questions)
    dans_corpus = [q for q in questions if q["source_attendue"].strip()]
    hors_corpus = [q for q in questions if not q["source_attendue"].strip()]

    # Context Precision@3 : le bon document est-il dans le top 3 ?
    corrects = 0
    for q in dans_corpus:
        passages = chercher(q["question"], k=3)
        sources = {p["source"] for p in passages}
        if q["source_attendue"] in sources:
            corrects += 1

    # Refus hors corpus : aucun passage ne doit passer le seuil
    refus = 0
    for q in hors_corpus:
        passages = chercher(q["question"], k=4)
        if not passages:
            refus += 1

    precision = corrects / len(dans_corpus) * 100 if dans_corpus else 0
    taux_refus = refus / len(hors_corpus) * 100 if hors_corpus else 0

    print(f"Seuil de distance : {SEUIL_DISTANCE}")
    print(f"Questions dans le corpus : {len(dans_corpus)}")
    print(f"  Context Precision@3 : {corrects}/{len(dans_corpus)} = {precision:.1f} %")
    print(f"Questions hors corpus : {len(hors_corpus)}")
    print(f"  Taux de refus : {refus}/{len(hors_corpus)} = {taux_refus:.1f} %")
    print(f"Total : {total} questions")

    return {
        "seuil": SEUIL_DISTANCE,
        "dans_corpus": len(dans_corpus),
        "corrects": corrects,
        "precision": precision,
        "hors_corpus": len(hors_corpus),
        "refus": refus,
        "taux_refus": taux_refus,
    }


if __name__ == "__main__":
    evaluer()
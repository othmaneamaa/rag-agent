#!/usr/bin/env python
"""Évalue la recherche sur les questions du corpus et les questions hors corpus.

Usage :
  python src/evaluate.py                    # PIAF complet : écrit models/metrics.json (versionné)
  python src/evaluate.py --corpus exemple   # l'extrait : écrit models/metrics-exemple.json

Chaque question de PIAF connaît le paragraphe dont elle est tirée : la recherche se mesure donc
sans annotation supplémentaire, comme le F1 de sentiment-app se mesure sur son jeu de test.
"""
import argparse
import json
from datetime import date

from communs import MODELS, argument_corpus
from mesurer import mesurer
from rag_agent import config, evaluation

SEUIL_RAPPEL_3 = 0.80  # en dessous, la recherche n'est pas assez bonne pour être livrée


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    argument_corpus(parser)
    args = parser.parse_args()

    m = mesurer(args.corpus)
    seuil = evaluation.au_seuil(config.SEUIL_DISTANCE, m.distances, m.rangs, m.distances_hors, config.TOP_K)
    metriques = {
        "source": m.source,
        "date": date.today().isoformat(),
        "modele_embedding": m.index.modele,
        "n_passages": m.n_passages,
        "n_questions": m.n_questions,
        "rappel_1": round(evaluation.rappel_a(1, m.rangs), 3),
        "rappel_3": round(evaluation.rappel_a(3, m.rangs), 3),
        "mrr_10": round(evaluation.mrr(m.rangs), 3),
        "top_k": config.TOP_K,
        "seuil_distance": config.SEUIL_DISTANCE,
        "repond": round(seuil["repond"], 3),
        "bon_passage_garde": round(seuil["bon_passage_garde"], 3),
        "n_hors_corpus": m.n_hors,
        "refus_hors_corpus": round(seuil["refus_hors_corpus"], 3),
        "seuil_rappel_3": SEUIL_RAPPEL_3,
    }

    print(f"Corpus {m.source} : {m.n_passages} passages, {m.n_questions} questions ({m.index.modele})")
    print(f"  rappel@1 {metriques['rappel_1']:.3f}   rappel@3 {metriques['rappel_3']:.3f}   "
          f"MRR@10 {metriques['mrr_10']:.3f}")
    print(f"Au seuil {config.SEUIL_DISTANCE} (top {config.TOP_K}) :")
    print(f"  questions du corpus auxquelles l'agent répond : {metriques['repond']:.1%}")
    print(f"  ... avec le bon passage parmi ceux gardés : {metriques['bon_passage_garde']:.1%}")
    print(f"  questions hors corpus refusées sans appel au LLM : {metriques['refus_hors_corpus']:.1%} "
          f"({m.n_hors} questions)")

    nom = "metrics.json" if m.source == "piaf" else f"metrics-{m.source}.json"
    MODELS.mkdir(exist_ok=True)
    (MODELS / nom).write_text(json.dumps(metriques, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Métriques : models/{nom}")

    if evaluation.rappel_a(3, m.rangs) < SEUIL_RAPPEL_3:
        print(f"Rappel@3 sous le seuil d'acceptation ({SEUIL_RAPPEL_3}) : la recherche n'est pas livrable.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

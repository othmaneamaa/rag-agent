#!/usr/bin/env python
"""La courbe du seuil de distance : combien de réponses, combien de refus hors corpus.

Usage :
  python src/courbe_seuil.py
  python src/courbe_seuil.py --corpus exemple

Pour chaque seuil, la part des questions du corpus auxquelles l'agent répond (au moins un passage
sous le seuil), la part où le bon passage est gardé, et la part des questions hors corpus refusées
sans appeler le modèle de langage. C'est le compromis de l'abstention de sentiment-app : un seuil
bas refuse tout ce qui est douteux, mais aussi des questions légitimes.
"""
import argparse

from communs import argument_corpus
from mesurer import mesurer
from rag_agent import config, evaluation

SEUILS = [0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    argument_corpus(parser)
    args = parser.parse_args()
    m = mesurer(args.corpus)
    print(f"Corpus {m.source} : {m.n_questions} questions du corpus, {m.n_hors} hors corpus "
          f"(top {config.TOP_K})\n")
    print(" seuil   répond   bon passage gardé   refus hors corpus")
    for seuil in SEUILS:
        r = evaluation.au_seuil(seuil, m.distances, m.rangs, m.distances_hors, config.TOP_K)
        marque = "  <- retenu" if abs(seuil - config.SEUIL_DISTANCE) < 1e-9 else ""
        print(f"  {seuil:.2f}   {r['repond']:6.1%}   {r['bon_passage_garde']:17.1%}   "
              f"{r['refus_hors_corpus']:17.1%}{marque}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

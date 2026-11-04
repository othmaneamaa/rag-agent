#!/usr/bin/env python
"""Ingestion : un passage par paragraphe du corpus, ses embeddings, l'index ChromaDB.

Usage :
  python src/ingest.py                    # corpus PIAF complet (python src/telecharger.py avant)
  python src/ingest.py --corpus exemple   # l'extrait versionné, comme la CI

L'index est écrit dans models/chroma (ou dans RAG_INDEX) et n'entre jamais dans Git :
il se reconstruit en une commande, comme le modèle de sentiment-app.
"""
import argparse
import time

from communs import argument_corpus, chemin_corpus
from rag_agent import config, corpus
from rag_agent.index import construire, encodeur_par_defaut


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    argument_corpus(parser)
    args = parser.parse_args()

    df = corpus.lire(chemin_corpus(args.corpus))
    passages = corpus.passages(df)
    print(f"Corpus {args.corpus} : {len(passages)} paragraphes, {df['titre'].nunique()} sujets")

    debut = time.time()
    index = construire(passages, config.chemin_index(), encodeur_par_defaut())
    print(f"Index : {config.chemin_index()} ({index.taille} passages, {index.modele})")
    print(f"Durée : {time.time() - debut:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

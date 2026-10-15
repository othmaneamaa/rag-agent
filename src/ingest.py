"""Ingestion : charge, découpe et indexe tous les PDF du dossier data/pdfs."""
from pathlib import Path
import sys

# Ajoute la racine du projet au chemin d'import
RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))

from rag.chunking import charger_documents, decouper
from rag.embeddings import charger_modele, encoder
from rag.vector_store import creer_collection, indexer


def main():
    print("1. Chargement des documents...")
    documents = charger_documents()
    print(f"   {len(documents)} pages chargées")

    if not documents:
        print("Aucun PDF trouvé dans data/pdfs/. Placez-y vos documents.")
        return

    print("2. Découpage en passages...")
    passages = decouper(documents)
    print(f"   {len(passages)} passages créés")

    print("3. Génération des embeddings...")
    modele = charger_modele()
    textes = [p.page_content for p in passages]
    vecteurs = encoder(textes, modele)
    print(f"   {len(vecteurs)} vecteurs générés")

    print("4. Indexation dans ChromaDB...")
    collection = creer_collection()
    indexer(collection, passages, vecteurs)
    print(f"   {collection.count()} passages indexés au total")


if __name__ == "__main__":
    main()
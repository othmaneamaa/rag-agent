"""Recherche les passages les plus proches d'une question."""
from pathlib import Path
import sys

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))


def chercher(question: str, k: int = 4):
    """Retourne les k passages les plus proches de la question."""
    if not question or not question.strip():
        raise ValueError("question vide")

    # Imports tardifs : évite de charger sentence-transformers au chargement
    from rag.embeddings import charger_modele, encoder
    from rag.vector_store import creer_collection, rechercher

    modele = charger_modele()
    collection = creer_collection()
    vecteur = encoder([question], modele)[0]
    resultats = rechercher(collection, vecteur, k=k)

    passages = []
    for doc, meta, dist in zip(
        resultats["documents"][0],
        resultats["metadatas"][0],
        resultats["distances"][0],
    ):
        passages.append({
            "texte": doc,
            "source": meta.get("source", "inconnu"),
            "page": meta.get("page", 0),
            "distance": float(dist),
        })
    return passages


if __name__ == "__main__":
    import sys as _sys
    question = " ".join(_sys.argv[1:]) or "Qu'est-ce que le machine learning ?"
    print(f"Question : {question}\n")
    for i, p in enumerate(chercher(question), 1):
        print(f"[{i}] {p['source']} (page {p['page']}, distance {p['distance']:.3f})")
        print(f"    {p['texte'][:200]}...")
        print()
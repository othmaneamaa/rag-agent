"""Base vectorielle ChromaDB persistante."""
import chromadb

from src.config import CHROMA

COLLECTION = "documents"


def creer_collection():
    """Crée (ou récupère) une collection persistante."""
    client = chromadb.PersistentClient(path=str(CHROMA))
    return client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def indexer(collection, passages, embeddings):
    """Ajoute les passages et leurs vecteurs à la collection."""
    if not passages:
        raise ValueError("aucun passage à indexer")

    collection.add(
        ids=[f"p{i}" for i in range(len(passages))],
        documents=[p.page_content for p in passages],
        metadatas=[
            {
                "source": p.metadata.get("source", "inconnu"),
                "page": p.metadata.get("page", 0),
            }
            for p in passages
        ],
        embeddings=embeddings.tolist(),
    )


def rechercher(collection, vecteur, k=4):
    """Retourne les k passages les plus proches du vecteur."""
    return collection.query(
        query_embeddings=[vecteur.tolist()],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
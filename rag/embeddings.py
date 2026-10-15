"""Génération des embeddings avec un modèle local (gratuit)."""
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def charger_modele() -> SentenceTransformer:
    """Le modèle est chargé une seule fois par processus."""
    return SentenceTransformer(EMBEDDING_MODEL)


def encoder(textes, modele=None):
    """Encode une liste de textes en vecteurs normalisés."""
    if modele is None:
        modele = charger_modele()
    return modele.encode(textes, normalize_embeddings=True, show_progress_bar=False)
"""L'index vectoriel : construire, ouvrir, chercher.

Les embeddings passent par un « encodeur » (textes -> vecteurs normés). Par défaut, c'est le
modèle sentence-transformers de config.MODELE_EMBEDDING, chargé une seule fois. Les tests en
injectent un autre, léger, pour tourner sans torch.

L'index retient le nom du modèle qui l'a construit : interroger un index avec un autre modèle
donnerait des distances sans aucun sens, comme relire un modèle scikit-learn avec une autre
version. ouvrir() refuse ce mélange.
"""
from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, TypedDict

import numpy as np

from rag_agent import config
from rag_agent.corpus import Passage


class Encodeur(Protocol):
    """Transforme des textes en vecteurs normés ; les questions et les passages peuvent différer."""

    nom: str

    def passages(self, textes: Sequence[str]) -> np.ndarray: ...

    def questions(self, textes: Sequence[str]) -> np.ndarray: ...


class Resultat(TypedDict):
    """Un passage retrouvé pour une question."""

    passage_id: str
    titre: str
    texte: str
    distance: float


@lru_cache(maxsize=2)
def _modele(nom: str) -> Any:
    from sentence_transformers import SentenceTransformer  # import tardif : torch est lourd

    return SentenceTransformer(nom)


class EncodeurSentenceTransformers:
    """Un modèle sentence-transformers, chargé une seule fois par processus.

    Les modèles E5 ont été entraînés avec un préfixe « query: » devant les questions et
    « passage: » devant les documents : sans eux, leur rappel chute.
    """

    def __init__(self, nom: str = config.MODELE_EMBEDDING) -> None:
        self.nom = nom
        e5 = "e5" in nom.lower()
        self._prefixe_question = "query: " if e5 else ""
        self._prefixe_passage = "passage: " if e5 else ""

    def _encoder(self, textes: Sequence[str]) -> np.ndarray:
        vecteurs = _modele(self.nom).encode(
            list(textes), normalize_embeddings=True, show_progress_bar=False, batch_size=64
        )
        return np.asarray(vecteurs, dtype=np.float32)

    def passages(self, textes: Sequence[str]) -> np.ndarray:
        return self._encoder([self._prefixe_passage + t for t in textes])

    def questions(self, textes: Sequence[str]) -> np.ndarray:
        return self._encoder([self._prefixe_question + t for t in textes])


def encodeur_par_defaut(nom: str = config.MODELE_EMBEDDING) -> Encodeur:
    return EncodeurSentenceTransformers(nom)


#: Recherche HNSW large : sur moins de quelques milliers de passages, elle devient quasi exacte et
#: reproductible. Avec les réglages par défaut (ef 10), ChromaDB 0.5 manquait des voisins : le
#: rappel@3 tombait de 0.832 à 0.803, et deux évaluations ne donnaient pas les mêmes chiffres.
REGLAGES_HNSW = {"hnsw:space": "cosine", "hnsw:construction_ef": 200, "hnsw:search_ef": 200, "hnsw:M": 32}


def _client(chemin: Path) -> Any:
    import logging

    import chromadb  # import tardif : l'API démarre plus vite, les tests légers s'en passent
    from chromadb.config import Settings

    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

    return chromadb.PersistentClient(path=str(chemin), settings=Settings(anonymized_telemetry=False))


class Index:
    """Une collection ChromaDB de passages, et l'encodeur qui l'interroge."""

    def __init__(self, collection: Any, encodeur: Encodeur) -> None:
        self.collection = collection
        self.encodeur = encodeur

    @property
    def taille(self) -> int:
        return int(self.collection.count())

    @property
    def modele(self) -> str:
        return str((self.collection.metadata or {}).get("modele", "inconnu"))

    def chercher(
        self, question: str, k: int = config.TOP_K, seuil: float | None = config.SEUIL_DISTANCE
    ) -> list[Resultat]:
        """Les k passages les plus proches ; ceux au-delà du seuil sont écartés (None : aucun filtre)."""
        return self.chercher_lot([question], k=k, seuil=seuil)[0]

    def chercher_lot(
        self, questions: Sequence[str], k: int = config.TOP_K, seuil: float | None = config.SEUIL_DISTANCE
    ) -> list[list[Resultat]]:
        """chercher() pour plusieurs questions d'un coup : un seul encodage, une seule requête."""
        for question in questions:
            if not isinstance(question, str) or not question.strip():
                raise ValueError("question vide")
        vecteurs = self.encodeur.questions(list(questions))
        lots: list[list[Resultat]] = []
        for debut in range(0, len(vecteurs), 200):  # ChromaDB limite la taille d'une requête
            brut = self.collection.query(
                query_embeddings=vecteurs[debut : debut + 200].tolist(),
                n_results=min(k, max(self.taille, 1)),
                include=["documents", "metadatas", "distances"],
            )
            for ids, textes, metas, distances in zip(
                brut["ids"], brut["documents"], brut["metadatas"], brut["distances"], strict=True
            ):
                lots.append([
                    {"passage_id": pid, "titre": str(meta.get("titre", "")), "texte": texte,
                     "distance": round(float(distance), 4)}
                    for pid, texte, meta, distance in zip(ids, textes, metas, distances, strict=True)
                    if seuil is None or distance <= seuil
                ])
        return lots


def construire(passages: Sequence[Passage], chemin: Path, encodeur: Encodeur) -> Index:
    """(Re)construit l'index : l'ancienne collection est remplacée, jamais complétée en double."""
    if not passages:
        raise ValueError("aucun passage à indexer")
    chemin.mkdir(parents=True, exist_ok=True)
    client = _client(chemin)
    if config.COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(config.COLLECTION)
    collection = client.create_collection(
        config.COLLECTION, metadata={**REGLAGES_HNSW, "modele": encodeur.nom}
    )
    vecteurs = encodeur.passages([p.texte for p in passages])
    for debut in range(0, len(passages), 500):  # ChromaDB limite la taille d'un ajout
        lot = passages[debut : debut + 500]
        collection.add(
            ids=[p.id for p in lot],
            documents=[p.texte for p in lot],
            metadatas=[{"titre": p.titre} for p in lot],
            embeddings=vecteurs[debut : debut + 500].tolist(),
        )
    return Index(collection, encodeur)


def ouvrir(chemin: Path | None = None, encodeur: Encodeur | None = None) -> Index:
    """Ouvre un index existant, ou explique comment le construire."""
    chemin = chemin or config.chemin_index()
    aide = "Lancez python src/ingest.py, ou indiquez son dossier dans la variable RAG_INDEX"
    if not chemin.exists():
        raise FileNotFoundError(f"index introuvable : {chemin}. {aide}")
    client = _client(chemin)
    if config.COLLECTION not in [c.name for c in client.list_collections()]:
        raise FileNotFoundError(f"index vide : {chemin}. {aide}")
    collection = client.get_collection(config.COLLECTION)
    modele = (collection.metadata or {}).get("modele", "inconnu")
    encodeur = encodeur or encodeur_par_defaut(modele)
    if encodeur.nom != modele:
        raise ValueError(
            f"l'index a été construit avec {modele}, pas avec {encodeur.nom} : "
            "reconstruisez-le avec python src/ingest.py"
        )
    return Index(collection, encodeur)

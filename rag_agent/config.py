"""Réglages de l'agent, lus dans les variables d'environnement (ou le fichier .env).

Comme SENTIMENT_MODELE dans sentiment-app, RAG_INDEX indique où trouver l'index vectoriel :
le paquet installé ne le contient pas, il se reconstruit avec « python src/ingest.py ».
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

RACINE = Path(__file__).resolve().parents[1]

#: Modèle d'embedding multilingue, choisi sur le corpus PIAF (voir docs/evaluation.md).
MODELE_EMBEDDING = os.getenv(
    "EMBEDDING_MODEL", "intfloat/multilingual-e5-small"
)
#: Modèle de langage servi par Groq.
MODELE_LLM = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
#: Nombre de passages récupérés par question.
TOP_K = int(os.getenv("TOP_K", "4"))
#: Au-delà de cette distance cosinus, un passage est jugé hors sujet (mesuré par src/courbe_seuil.py).
SEUIL_DISTANCE = float(os.getenv("SEUIL_DISTANCE", "0.19"))
#: Nom de la collection ChromaDB.
COLLECTION = "piaf"


def chemin_index() -> Path:
    """RAG_INDEX si elle est définie, sinon l'index construit dans le dépôt."""
    return Path(os.environ.get("RAG_INDEX") or RACINE / "models" / "chroma")

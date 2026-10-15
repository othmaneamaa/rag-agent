"""Configuration centrale du projet RAG."""
import os
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
DATA = RACINE / "data"
PDFS = DATA / "pdfs"
ARTICLES = DATA / "articles"
MODELS = RACINE / "models"
CHROMA = MODELS / "chroma"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
TOP_K = int(os.getenv("TOP_K", 4))
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

for d in (PDFS, ARTICLES, CHROMA):
    d.mkdir(parents=True, exist_ok=True)
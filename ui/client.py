"""Client HTTP de l'API : tout ce que l'interface sait de l'agent passe par ce module.

L'interface n'ouvre jamais l'index et n'appelle jamais Groq : elle envoie des requêtes à l'API
et traduit chaque réponse en message pour l'utilisateur. Testable sans Streamlit.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import httpx

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
INJOIGNABLE = "API injoignable : lancez d'abord rag serve (ou uvicorn api.main:app)."
REFUSEE = "Question refusée : vide, faite d'espaces, ou de plus de 500 caractères."
SANS_INDEX = "Index non chargé : python src/ingest.py, puis relancer l'API."
SANS_LLM = "Rédaction indisponible : la clé Groq manque dans .env. La recherche seule fonctionne."
GENERATION = "Le service de génération ne répond pas : réessayez dans un instant."


@dataclass
class Reponse:
    """Ce que l'interface affiche : un résultat, ou un message d'erreur lisible."""

    ok: bool
    message: str = ""
    corps: dict[str, Any] = field(default_factory=dict)

    @property
    def refus(self) -> bool:
        return self.ok and self.corps.get("statut") == "REFUS"


def creer_session() -> httpx.Client:
    """La connexion à l'API. Les tests la remplacent par un client en mémoire."""
    return httpx.Client(base_url=API_URL, timeout=60)


def _traduire(r: httpx.Response) -> Reponse:
    if r.status_code == 200:
        return Reponse(True, corps=r.json())
    if r.status_code == 422:
        return Reponse(False, REFUSEE)
    if r.status_code == 503:
        detail = str(r.json().get("detail", "")) if r.headers.get("content-type", "").startswith("application/json") else ""
        return Reponse(False, SANS_LLM if "Groq" in detail else SANS_INDEX)
    if r.status_code == 502:
        return Reponse(False, GENERATION)
    return Reponse(False, f"Erreur interne de l'API ({r.status_code}).")


def _appeler(session: httpx.Client, methode: str, chemin: str, **kwargs: Any) -> Reponse:
    try:
        return _traduire(session.request(methode, chemin, **kwargs))
    except httpx.TransportError:
        return Reponse(False, INJOIGNABLE)


def sante(session: httpx.Client) -> Reponse:
    return _appeler(session, "GET", "/health")


def demander(session: httpx.Client, question: str, k: int = 4) -> Reponse:
    return _appeler(session, "POST", "/ask", json={"question": question, "k": k})


def chercher(session: httpx.Client, question: str, k: int = 4) -> Reponse:
    return _appeler(session, "POST", "/search", json={"question": question, "k": k})


def citation(source: dict[str, Any]) -> str:
    return f"{source['titre']} · {source['passage_id']} · distance {source['distance']:.3f}"

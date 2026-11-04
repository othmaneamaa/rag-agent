"""Le modèle de langage, servi par Groq. La clé reste dans .env, jamais dans le code ni dans Git."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from rag_agent import config


class LLMNonConfigure(RuntimeError):
    """La clé GROQ_API_KEY manque : la recherche marche, la rédaction non."""


def configure() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


@lru_cache(maxsize=1)
def _client() -> Any:
    if not configure():
        raise LLMNonConfigure(
            "GROQ_API_KEY absente : copiez .env.example en .env et renseignez votre clé Groq"
        )
    from groq import Groq  # import tardif : inutile pour la recherche seule

    return Groq(api_key=os.environ["GROQ_API_KEY"])


def generer(prompt: str, temperature: float = 0.1) -> str:
    """Envoie le prompt à Groq et renvoie le texte de la réponse."""
    completion = _client().chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=config.MODELE_LLM,
        temperature=temperature,
    )
    return str(completion.choices[0].message.content or "").strip()

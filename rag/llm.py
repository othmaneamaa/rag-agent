"""Wrapper LLM Groq."""
import os
from functools import lru_cache

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


@lru_cache(maxsize=1)
def get_client() -> Groq:
    """Client Groq, instancié une seule fois."""
    cle = os.getenv("GROQ_API_KEY")
    if not cle:
        raise ValueError(
            "GROQ_API_KEY absente. Créez un fichier .env à la racine du projet."
        )
    return Groq(api_key=cle)


def generer(prompt: str, model: str | None = None, temperature: float = 0.1) -> str:
    """Envoie un prompt à Groq et retourne le texte de la réponse."""
    model = model or os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
    client = get_client()
    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=temperature,
    )
    return completion.choices[0].message.content
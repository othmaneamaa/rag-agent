"""Contrats d'entrée et de sortie de l'API (Pydantic v2)."""
from typing import Literal

from pydantic import BaseModel, Field, field_validator

LONGUEUR_MAX = 500


class QuestionRequest(BaseModel):
    """Une question en langage naturel."""

    question: str = Field(
        ..., min_length=1, max_length=LONGUEUR_MAX, description="La question à poser au corpus",
        examples=["Combien d'habitants compte Moscou ?"],
    )
    k: int = Field(4, ge=1, le=10, description="Nombre de passages à récupérer")

    @field_validator("question")
    @classmethod
    def question_non_vide(cls, valeur: str) -> str:
        """Refuse une question faite uniquement de blancs (même règle que Index.chercher)."""
        if not valeur.strip():
            raise ValueError("question vide")
        return valeur


class Passage(BaseModel):
    passage_id: str = Field(..., description="Identifiant stable du paragraphe, cité dans la réponse")
    titre: str = Field(..., description="Titre de l'article Wikipédia")
    texte: str
    distance: float = Field(..., ge=0, le=2, description="Distance cosinus à la question")


class Source(BaseModel):
    passage_id: str
    titre: str
    distance: float


class SearchResponse(BaseModel):
    question: str
    seuil: float = Field(..., description="Distance au-delà de laquelle un passage est écarté")
    passages: list[Passage]


class AskResponse(BaseModel):
    statut: Literal["REPONSE", "REFUS"] = Field(
        ..., description="REFUS : rien de pertinent dans le corpus, l'agent n'invente pas"
    )
    reponse: str
    sources: list[Source]
    passages: list[Passage]


class HealthResponse(BaseModel):
    status: str = Field(..., description="OK si l'index est chargé")
    passages: int = Field(..., description="Nombre de passages indexés")
    modele_embedding: str
    llm: str = Field(..., description="Modèle de langage, ou « non configuré » sans clé Groq")
    seuil: float

"""API REST de l'agent de recherche documentaire.

Lancer : uvicorn api.main:app --reload   (ou : rag serve)
Doc    : http://localhost:8000/docs
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.agent import Agent, charger_agent
from api.schemas import AskResponse, HealthResponse, QuestionRequest, SearchResponse
from rag_agent import __version__, config
from rag_agent.llm import LLMNonConfigure

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rag_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ouvre l'index une seule fois, avant la première requête."""
    app.state.agent = charger_agent()
    logger.info("Index chargé : %s passages (%s)", app.state.agent.index.taille, app.state.agent.index.modele)
    yield


app = FastAPI(
    title="RAG Agent API",
    description="Répond aux questions sur le corpus PIAF (Wikipédia en français), avec ses sources.",
    version=__version__,
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _agent() -> Agent:
    agent = getattr(app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Index non chargé")
    return agent


@app.get("/", summary="Accueil")
async def accueil():
    return {"message": "RAG Agent API", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse, summary="Santé de l'API")
async def health():
    agent = _agent()
    return {
        "status": "OK",
        "passages": agent.index.taille,
        "modele_embedding": agent.index.modele,
        "llm": agent.nom_llm,
        "seuil": config.SEUIL_DISTANCE,
    }


@app.post("/search", response_model=SearchResponse, summary="Chercher des passages, sans rédiger")
def search(requete: QuestionRequest):
    """Les passages les plus proches sous le seuil. Ne demande pas de clé Groq."""
    agent = _agent()
    try:
        passages = agent.chercher(requete.question, requete.k)
    except Exception:
        logger.exception("Erreur pendant la recherche")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur") from None
    return {"question": requete.question, "seuil": config.SEUIL_DISTANCE, "passages": passages}


@app.post("/ask", response_model=AskResponse, summary="Poser une question")
def ask(requete: QuestionRequest):
    """Une réponse rédigée et sourcée, ou un refus.

    - 200, statut REFUS : aucun passage pertinent, ou le modèle de langage ne trouve pas la réponse
    - 422 : question vide, faite d'espaces ou de plus de 500 caractères
    - 503 : index non chargé, ou clé Groq absente
    - 502 : le service de génération a échoué (détail journalisé, jamais renvoyé au client)
    """
    agent = _agent()
    try:
        resultat = agent.repondre(requete.question, requete.k)
    except LLMNonConfigure:
        raise HTTPException(status_code=503, detail="Génération indisponible : clé Groq absente") from None
    except Exception:
        logger.exception("Erreur pendant la génération")
        raise HTTPException(status_code=502, detail="Service de génération indisponible") from None
    logger.info("ask: %r -> %s (%d sources)", requete.question[:60], resultat["statut"], len(resultat["sources"]))
    return resultat

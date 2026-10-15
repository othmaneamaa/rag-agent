"""Logique de l'interface, testable sans Streamlit."""
from pathlib import Path
import sys

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))


def charger_agent():
    """Import tardif : n'a de sens qu'avec le modèle et la clé API présents."""
    from src.generate import repondre
    return repondre


def poser_question(question, agent):
    """Valide la question et appelle l'agent."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question vide")
    resultat = agent(question)
    # Déduplique les sources tout en gardant l'ordre
    vues = set()
    uniques = []
    for s in resultat.get("sources", []):
        cle = (s["source"], s["page"])
        if cle not in vues:
            vues.add(cle)
            uniques.append(s)
    resultat["sources_uniques"] = uniques
    return resultat


def formater_citation(source):
    """Retourne une citation lisible."""
    return f"{source['source']} — page {source['page']}"
"""Génère une réponse ancrée dans les passages récupérés."""
from pathlib import Path
import sys

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))

# Imports tardifs : n'ont de sens qu'à l'appel, pas au chargement
def _generer(*args, **kwargs):
    from rag.llm import generer
    return generer(*args, **kwargs)

def _chercher(*args, **kwargs):
    from src.retrieve import chercher
    return chercher(*args, **kwargs)


PROMPT = """Tu es un assistant documentaire. Réponds à la question en te basant
UNIQUEMENT sur les extraits fournis ci-dessous. Si la réponse ne s'y trouve pas,
dis-le clairement. Pour chaque affirmation, cite la source entre crochets,
au format [source, page].

Extraits :
{extraits}

Question : {question}

Réponse :"""


def formater_extraits(passages):
    lignes = []
    for i, p in enumerate(passages, 1):
        lignes.append(
            f"[Extrait {i}] ({p['source']}, page {p['page']})\n{p['texte']}"
        )
    return "\n\n".join(lignes)


def repondre(question: str, k: int = 4):
    if not question or not question.strip():
        raise ValueError("question vide")

    passages = _chercher(question, k=k)
    if not passages:
        return {
            "question": question,
            "reponse": "Aucun document pertinent trouvé.",
            "sources": [],
            "passages": [],
        }

    prompt = PROMPT.format(
        extraits=formater_extraits(passages),
        question=question,
    )
    reponse = _generer(prompt)

    sources = sorted({(p["source"], p["page"]) for p in passages})
    return {
        "question": question,
        "reponse": reponse,
        "sources": [{"source": s, "page": p} for s, p in sources],
        "passages": passages,
    }


if __name__ == "__main__":
    import sys as _sys
    question = " ".join(_sys.argv[1:]) or "Qu'est-ce que le RAG ?"
    resultat = repondre(question)
    print(f"Question : {resultat['question']}\n")
    print("Réponse :")
    print(resultat["reponse"])
    print()
    print("Sources :")
    for i, s in enumerate(resultat["sources"], 1):
        print(f"  [{i}] {s['source']} — page {s['page']}")
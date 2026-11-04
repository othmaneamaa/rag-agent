"""La commande « rag » : le paquet s'utilise sans connaître ses fichiers.

  rag --version
  rag search "Quand Lavoisier est-il né ?"     (sans clé Groq)
  rag ask "Quand Lavoisier est-il né ?"        (avec GROQ_API_KEY)
  rag ingest chemin/vers/train.csv
  rag serve [--host 127.0.0.1] [--port 8000]
  rag ui
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from rag_agent import __version__


def _index():
    from rag_agent.index import ouvrir

    return ouvrir()


def _chercher(args: argparse.Namespace) -> int:
    from rag_agent import config

    try:
        passages = _index().chercher(" ".join(args.question), k=args.k, seuil=config.SEUIL_DISTANCE)
    except FileNotFoundError as erreur:
        print(erreur, file=sys.stderr)
        return 1
    if not passages:
        print(f"Aucun passage sous le seuil de {config.SEUIL_DISTANCE} : l'agent refuserait.")
    for p in passages:
        print(f"[{p['titre']}, {p['passage_id']}] distance {p['distance']:.3f}\n  {p['texte'][:200]}...")
    return 0


def _demander(args: argparse.Namespace) -> int:
    from rag_agent import llm
    from rag_agent.generation import repondre

    try:
        resultat = repondre(" ".join(args.question), _index(), llm.generer, k=args.k)
    except (FileNotFoundError, llm.LLMNonConfigure) as erreur:
        print(erreur, file=sys.stderr)
        return 1
    print(json.dumps(
        {"statut": resultat["statut"], "reponse": resultat["reponse"], "sources": resultat["sources"]},
        ensure_ascii=False, indent=2,
    ))
    return 0


def _ingerer(args: argparse.Namespace) -> int:
    from rag_agent import config, corpus
    from rag_agent.index import construire, encodeur_par_defaut

    passages = corpus.passages(corpus.lire(Path(args.csv)))
    index = construire(passages, config.chemin_index(), encodeur_par_defaut())
    print(f"{index.taille} passages indexés dans {config.chemin_index()}")
    return 0


def _servir(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("api.main:app", host=args.host, port=args.port)
    return 0


def _interface(args: argparse.Namespace) -> int:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print('Streamlit absent : pip install "rag-agent[ui]"', file=sys.stderr)
        return 1
    page = Path(__file__).resolve().parents[1] / "ui" / "app.py"
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(page)])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag", description="Agent de recherche documentaire (PIAF).")
    parser.add_argument("--version", action="version", version=f"rag-agent {__version__}")
    commandes = parser.add_subparsers(dest="commande", required=True)

    for nom, aide, fonction in [
        ("search", "cherche les passages, sans rédiger", _chercher),
        ("ask", "répond avec ses sources (clé Groq nécessaire)", _demander),
    ]:
        commande = commandes.add_parser(nom, help=aide)
        commande.add_argument("question", nargs="+")
        commande.add_argument("-k", type=int, default=4, help="passages récupérés")
        commande.set_defaults(fonction=fonction)

    ingerer = commandes.add_parser("ingest", help="indexe un fichier au format PIAF")
    ingerer.add_argument("csv")
    ingerer.set_defaults(fonction=_ingerer)

    servir = commandes.add_parser("serve", help="lance l'API")
    servir.add_argument("--host", default="127.0.0.1")
    servir.add_argument("--port", type=int, default=8000)
    servir.set_defaults(fonction=_servir)

    interface = commandes.add_parser("ui", help="lance la page Streamlit (l'API doit tourner)")
    interface.set_defaults(fonction=_interface)

    args = parser.parse_args(argv)
    return int(args.fonction(args))

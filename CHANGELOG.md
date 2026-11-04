# Journal des versions

Chaque version publiée porte une étiquette Git `vX.Y.Z`. La version est écrite à un seul endroit : `rag_agent/__init__.py`.

## 0.2.0 : le corpus Kaggle, l'API, le paquet

- **Pour celui qui pose des questions** : le corpus devient le jeu Piaf (French Questions & Answers) de Kaggle, 761 paragraphes de Wikipédia en français sur 191 sujets, à la place de 5 PDF qui n'étaient pas dans le dépôt. Les réponses citent leurs passages au format `[titre, identifiant]`. L'agent refuse quand aucun passage n'est sous le seuil de distance (0,19), sans appeler Groq, ou quand Groq répond `HORS_CORPUS`.
- **Pour celui qui développe** : le code suit l'architecture de sentiment-app. Le paquet `rag_agent` (corpus, index, génération, mesures) est installable (`pip install -e .`) et fournit la commande `rag` : `--version`, `search`, `ask`, `ingest`, `serve`, `ui`. L'API FastAPI (`/health`, `/search`, `/ask`) a un contrat Pydantic et des codes d'erreur sans fuite d'information ; la page Streamlit passe par l'API. Plus aucun `sys.path` dans le code installé. Les scripts de `src/` téléchargent PIAF, indexent, évaluent et tracent la courbe du seuil.
- **Pour la mesure** : 3 835 questions évaluent la recherche (`models/metrics.json`, versionné) : rappel@3 0,832, contre 0,481 avec le modèle de la 0.1.0. Le modèle d'embedding passe de `all-MiniLM-L6-v2` (surtout anglais) à `intfloat/multilingual-e5-small`, choisi parmi quatre candidats (`docs/evaluation.md`). La recherche HNSW est élargie : avec les réglages par défaut, elle manquait des voisins et n'était pas reproductible.
- **Pour la CI** : versions épinglées, ruff, mypy, couverture (seuil 85 %), 91 tests dont 4 sur le vrai modèle ; l'extrait versionné est indexé et évalué ; la roue est construite, installée hors du dépôt et essayée.
- Corrigé : `ingest.py` ajoutait les passages à chaque lancement, l'index doublait ; il est maintenant reconstruit. Accents réparés dans les commentaires.
- Retiré : le chargement des PDF (`rag/chunking.py`), devenu inutile, et `data/questions_reference.csv`, qui portait sur les PDF.

## 0.1.0 : le RAG sur PDF (étiquette `rendu-final`)

- Ingestion de 5 PDF (découpage en passages de 512 caractères, `all-MiniLM-L6-v2`, ChromaDB), génération sourcée avec Groq, refus au seuil de 0,55, interface Streamlit, 15 tests, 20 questions de référence.

# Compte rendu : agent de recherche documentaire (RAG)

**Binôme :** Othmane Amaadour (@othmaneamaa) & Mohcine Errachid (@mohcine00errachind)
**Dépôt :** https://github.com/othmaneamaa/rag-agent
**Version :** 0.2.0 (étiquette `v0.2.0`) ; la version 0.1.0 reste sous l'étiquette `rendu-final`

---

## 1. Ce qui a changé depuis la 0.1.0, et pourquoi

La 0.1.0 indexait 5 PDF personnels, absents du dépôt : personne d'autre ne pouvait reconstruire l'index, rejouer l'évaluation, ni faire passer les tests de recherche, que la CI ignorait. La 0.2.0 reprend le projet sur deux bases :

- **un corpus public et reproductible** : le jeu Piaf (French Questions & Answers) de Kaggle, licence CC0, téléchargé en une commande ;
- **l'architecture de sentiment-app**, le projet de cours : données, scripts, paquet, API, interface, tests, CI, roue.

## 2. Le corpus

761 paragraphes de Wikipédia en français, sur 191 sujets, et 3 835 questions écrites par des humains, chacune sur un paragraphe précis. Un paragraphe fait de 500 à 996 caractères : on l'indexe tel quel, sans le découper, ce qui garde chaque citation lisible et sans phrase coupée.

Les questions changent la nature de l'évaluation. En 0.1.0, nous avions écrit 20 questions à la main ; ici, chaque question connaît son paragraphe, et la recherche se mesure sur 3 835 cas, comme le F1 de sentiment-app sur son jeu de test.

Le corpus complet reste hors de Git (`data/piaf/`), comme les données de sentiment-app ; un extrait de 10 sujets (81 paragraphes, 417 questions) est versionné pour les tests et la CI, ce que la licence CC0 permet.

## 3. Architecture

| Couche | Dossier | Rôle |
| --- | --- | --- |
| Données | `data/` | Extrait versionné, 30 questions hors corpus ; PIAF complet hors Git |
| Scripts | `src/` | `telecharger.py`, `corpus_exemple.py`, `ingest.py`, `evaluate.py`, `courbe_seuil.py` |
| Paquet | `rag_agent/` | `corpus`, `index`, `generation`, `llm`, `evaluation`, `config`, la commande `rag` |
| Service | `api/` | FastAPI : `/health`, `/search`, `/ask` |
| Interface | `ui/` | Streamlit et son client HTTP |

Trois règles reprises de sentiment-app : l'API ne construit rien, elle ouvre l'index au démarrage et refuse de démarrer sans lui ; l'index ne va jamais dans Git, il se reconstruit ; la page passe par l'API, jamais par l'index.

| Artefact | Dans Git ? | Raison |
| --- | --- | --- |
| Le code, les tests, la doc | Oui | C'est le travail |
| L'extrait du corpus, les questions hors corpus | Oui | Petits, écrits ou choisis par nous, nécessaires à la CI |
| `models/metrics.json` | Oui | C'est la preuve de la mesure |
| PIAF complet, l'index, `.venv/`, `.env` | Non | Se reconstruisent, ou sont secrets |

## 4. Les deux choix mesurés

**Le modèle d'embedding.** Sur les 3 835 questions, le modèle de la 0.1.0 (`all-MiniLM-L6-v2`) ne retrouve le bon paragraphe dans les 3 premiers que pour 48,1 % des questions : moins bien qu'une recherche par mots-clés TF-IDF (76,3 %), parce qu'il a été entraîné surtout sur de l'anglais. Nous avons comparé quatre options et retenu `intfloat/multilingual-e5-small` : rappel@3 **0,832**, MRR@10 0,757 (`docs/evaluation.md`).

**Le seuil de refus.** Pour 90 % des questions du corpus, le passage le plus proche est entre 0,106 et 0,190 de distance ; pour les questions hors corpus, entre 0,166 et 0,226. Les deux se chevauchent. À **0,19**, l'agent répond à 95,1 % des questions légitimes et refuse deux tiers des questions hors corpus sans appeler Groq. Les autres rencontrent un second refus : le prompt impose `HORS_CORPUS` quand les extraits ne contiennent pas la réponse, et le code transforme ce mot en refus, au lieu de laisser le modèle formuler une excuse que l'interface ne saurait pas reconnaître.

## 5. Tests et qualité

91 tests, en deux familles. La logique (seuil, refus, citations, contrat de l'API, codes d'erreur, client de l'interface, commande `rag`) est testée avec un encodeur de hachage et un faux modèle de langage : sans torch, sans clé Groq, en quelques secondes. Le vrai modèle est testé sur l'extrait versionné (`tests/test_modele.py`). La CI exécute tout, avec ruff, mypy et la couverture, puis construit la roue et l'essaie hors du dépôt.

## 6. Ce qui a échoué en route

1. **Le modèle d'embedding de la 0.1.0 était inadapté au français** : 0,481 au rappel@3, sous TF-IDF. Nous ne l'aurions pas vu sans les 3 835 questions de PIAF.
2. **La recherche approchée de ChromaDB manquait des voisins.** Avec les réglages par défaut, le rappel@3 tombait de 0,832 à 0,803, et deux évaluations ne donnaient pas les mêmes chiffres. Corrigé en élargissant la recherche HNSW (`hnsw:search_ef` = 200).
3. **L'ingestion doublait l'index** : `ingest.py` ajoutait les passages sans vider la collection. Corrigé : l'index est reconstruit.
4. **Le seuil se lisait à l'import** : changer le réglage n'avait pas d'effet sur `repondre()`. Un test de la commande `rag ask` l'a révélé.
5. **Kaggle et Hugging Face bloqués** depuis notre environnement de travail : il a fallu autoriser leurs domaines. Le téléchargement d'un jeu public de Kaggle, lui, ne demande pas de compte.

## 7. Limites

Les questions qui ne se comprennent qu'avec leur paragraphe (9,1 % ne le retrouvent pas dans les 10 premiers), les questions courtes, et les sujets aux paragraphes voisins (batailles, sièges) sont les moins bien servis. Le refus est mesuré sur 30 questions seulement, et la fidélité des réponses de Groq n'est pas mesurée automatiquement. Le détail est dans `docs/evaluation.md` et `docs/fiche-modele.md`.

## 8. Ce qu'on ferait ensuite

Un reranker sur les 10 premiers passages, une recherche hybride TF-IDF et embeddings, un jeu hors corpus plus grand, et une mesure de la fidélité des réponses générées. Chacun se mesure avec `src/evaluate.py`.

## 9. Répartition du travail

| Membre | Contributions |
| --- | --- |
| **Othmane Amaadour** | *À compléter par le binôme* |
| **Mohcine Errachid** | *À compléter par le binôme* |

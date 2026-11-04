# Agent de recherche documentaire (RAG)

Un agent qui répond en français à des questions sur un corpus de documents, **en citant ses sources**, et qui **refuse** quand le corpus ne contient pas la réponse. Il suit l'architecture du projet de cours `sentiment-app` : un paquet installable, une API FastAPI, une page Streamlit qui passe par l'API, des tests, une CI.

**Corpus** : [Piaf (French Questions & Answers)](https://www.kaggle.com/datasets/thedevastator/unlock-the-power-of-reading-comprehension-with-p), sur Kaggle, licence CC0 : **761 paragraphes** de Wikipédia en français, sur **191 sujets**, et **3 835 questions** écrites par des humains sur ces paragraphes (projet PIAF d'Etalab). Les paragraphes forment le corpus ; les questions servent à mesurer la recherche, chacune connaissant le paragraphe dont elle est tirée.

| Mesure, sur les 3 835 questions (`models/metrics.json`) | Valeur |
| --- | --- |
| Bon paragraphe en 1re position (rappel@1) | 66,8 % |
| Bon paragraphe dans les 3 premiers (rappel@3) | **83,2 %** |
| MRR@10 | 0,757 |
| Questions du corpus auxquelles l'agent répond, au seuil 0,19 | 95,1 % |
| Questions hors corpus refusées sans appeler le LLM | 66,7 % (sur 30) |

## Architecture

```
Question ──► embedding (intfloat/multilingual-e5-small)
         ──► ChromaDB : les 4 passages les plus proches, distance cosinus ≤ 0,19
             │
             ├─ aucun passage ─────────────────────────► REFUS (sans appel au LLM)
             └─ passages ─► prompt « uniquement ces extraits »
                          ─► LLM Groq (openai/gpt-oss-120b)
                             ├─ « HORS_CORPUS » ───────► REFUS
                             └─ réponse + citations [titre, identifiant]
```

| Couche | Dossier | Ce qu'elle fait | Ce qu'elle ne fait jamais |
| --- | --- | --- | --- |
| Données | `data/` | L'extrait versionné `corpus_exemple.csv`, les 30 questions hors corpus ; PIAF complet dans `data/piaf/` (hors Git) | Aucun calcul |
| Scripts | `src/` | Télécharger, extraire l'exemple, indexer, évaluer, tracer la courbe du seuil | Servir des requêtes |
| Le paquet | `rag_agent/` | Corpus, index, recherche, génération sourcée, refus, mesures, commande `rag` | Connaître HTTP |
| Le service | `api/` | `/health`, `/search`, `/ask`, le contrat Pydantic, les codes d'erreur | Construire l'index |
| L'interface | `ui/` | La page et son client HTTP | Ouvrir l'index ou appeler Groq |

## Installation

Python 3.11, et une [clé Groq](https://console.groq.com/keys) pour la rédaction (la recherche seule s'en passe).

```powershell
git clone https://github.com/othmaneamaa/rag-agent.git
cd rag-agent
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e .
Copy-Item .env.example .env      # puis renseigner GROQ_API_KEY dans .env
```

## Utilisation

```powershell
python src/telecharger.py         # PIAF depuis Kaggle -> data/piaf/train.csv (aucun compte nécessaire)
python src/ingest.py              # l'index -> models/chroma (une minute, le modèle se télécharge la 1re fois)
python src/evaluate.py            # les mesures -> models/metrics.json
python src/courbe_seuil.py        # le compromis réponses / refus selon le seuil

rag search "Quand Antoine Lavoisier est-il né ?"   # les passages, sans LLM
rag ask "Quand Antoine Lavoisier est-il né ?"      # la réponse sourcée
rag serve                                          # l'API : http://127.0.0.1:8000/docs
rag ui                                             # la page, dans un second terminal
```

Sans télécharger PIAF : `python src/ingest.py --corpus exemple` indexe l'extrait versionné (10 sujets, 81 paragraphes).

### L'API

| Route | Rôle | Codes |
| --- | --- | --- |
| `GET /health` | Passages indexés, modèles, seuil | 200 ; 503 sans index |
| `POST /search` | `{"question": "...", "k": 4}` : les passages sous le seuil, sans LLM | 200 ; 422 hors contrat |
| `POST /ask` | La réponse et ses sources, ou `statut: "REFUS"` | 200 ; 422 ; 503 sans clé Groq ; 502 si Groq échoue, sans message interne |

Une question est acceptée si elle fait de 1 à 500 caractères et n'est pas faite que de blancs ; `k` va de 1 à 10.

## Tests et qualité

```powershell
python -m pytest -q                 # 91 tests
python -m pytest -q -m "not modele" # sans le vrai modèle d'embedding : quelques secondes
ruff check .
mypy rag_agent api ui src
```

Les tests de la logique utilisent un encodeur de hachage et un faux modèle de langage : ni torch, ni clé Groq. `tests/test_modele.py` vérifie le vrai modèle sur l'extrait versionné. La CI rejoue tout à chaque push, indexe l'extrait, l'évalue, puis construit la roue et l'essaie hors du dépôt.

## Le paquet

```powershell
python -m build --wheel
pip install dist\rag_agent-0.2.0-py3-none-any.whl
$env:RAG_INDEX = "C:\chemin\vers\models\chroma"
rag search "Quand Antoine Lavoisier est-il né ?"
```

La roue contient `rag_agent`, `api` et `ui`, ni les données, ni l'index, ni les tests : l'index se reconstruit, et le paquet le trouve par la variable `RAG_INDEX`.

## Documentation

- [`docs/cadrage/agent-recherche.md`](docs/cadrage/agent-recherche.md) : le besoin et le choix du RAG
- [`docs/evaluation.md`](docs/evaluation.md) : le choix du modèle d'embedding, du seuil, et les limites mesurées
- [`docs/fiche-modele.md`](docs/fiche-modele.md) : la fiche de l'agent, usages et conditions d'usage
- [`docs/compte-rendu.md`](docs/compte-rendu.md) : le compte rendu du binôme
- [`CHANGELOG.md`](CHANGELOG.md) : le journal des versions

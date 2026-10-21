# 🔍 Agent de recherche documentaire (RAG)

Agent capable d'interroger en **langage naturel** un corpus de PDF et de répondre avec des **sources citées**. Basé sur le pattern **RAG** (Retrieval-Augmented Generation).

---

## Fonctionnalités

- **Ingestion** : découpage des PDF en passages, embeddings via `sentence-transformers`, indexation dans ChromaDB.
- **Recherche** : retrieval vectoriel avec seuil de similarité (0.55) pour filtrer les passages hors sujet.
- **Génération** : LLM Groq (`openai/gpt-oss-120b`) qui rédige une réponse **ancrée dans les extraits**, avec citations.
- **Refus hors corpus** : si aucun passage ne passe le seuil, l'agent refuse de répondre (pas d'hallucination).
- **Interface Streamlit** : onglets Question, Corpus, Évaluation, Aide.
- **Évaluation quantitative** : 20 questions de référence, Context Precision@3, taux de refus.

---

## Architecture

```
Question
   │
   ▼
Embedding (sentence-transformers/all-MiniLM-L6-v2)
   │
   ▼
Recherche vectorielle (ChromaDB, top-k, seuil 0.55)
   │
   ▼
Prompt + extraits
   │
   ▼
LLM Groq (openai/gpt-oss-120b)
   │
   ▼
Réponse + citations
```

| Dossier | Rôle |
|---------|------|
| `rag/` | Bibliothèque interne (chunking, embeddings, vector store, LLM) |
| `src/` | Scripts (ingestion, retrieval, génération, évaluation) |
| `ui/` | Interface Streamlit |
| `tests/` | Tests pytest (15) |
| `data/pdfs/` | Corpus (5 PDF, non versionnés) |
| `models/chroma/` | Index vectoriel (non versionné) |
| `docs/` | Cadrage, évaluation, compte rendu |

---

## Installation

**Prérequis :** Python 3.11, une clé API Groq.

```powershell
# 1. Cloner le dépôt
git clone https://github.com/othmaneamaa/rag-agent.git
cd rag-agent

# 2. Créer l'environnement virtuel
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer la clé API
Copy-Item .env.example .env
notepad .env   # renseigner GROQ_API_KEY=gsk_...

---
rag-agent/
├── data/
│   ├── pdfs/                  # corpus (non versionné)
│   └── questions_reference.csv
├── docs/
│   ├── cadrage/agent-recherche.md
│   ├── evaluation.md
│   └── compte-rendu.md
├── rag/
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   └── llm.py
├── src/
│   ├── config.py
│   ├── ingest.py
│   ├── retrieve.py
│   ├── generate.py
│   └── evaluate_rag.py
├── ui/
│   ├── app.py
│   └── helpers.py
├── tests/
│   ├── conftest.py
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   ├── test_generation.py
│   └── test_ui.py
├── models/chroma/             # index vectoriel (non versionné)
├── .env.example
├── .gitignore
├── .streamlit/config.toml
├── requirements.txt
├── pytest.ini
└── README.md

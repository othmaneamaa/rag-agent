# Compte rendu — Agent de recherche documentaire (RAG)

**Binôme :** Othmane Amaadour (@othmaneamaa) & Mohcine Errachid (@mohcine00errachind)
**Dépôt :** https://github.com/othmaneamaa/rag-agent
**Date de remise :** 16 octobre 2026
**Étiquette Git :** `rendu-final`

---

## 1. Cadrage et besoin

### Le besoin métier

Permettre à un utilisateur d'interroger en **langage naturel** un corpus
hétérogène de documents (PDF, articles, pages internes) et d'obtenir des
réponses **sourcées**, c'est-à-dire accompagnées des passages exacts dont
elles sont tirées.

### La grille de décision

Nous avons appliqué la grille du cours pour choisir l'approche :

| Critère | Réponse | Commentaire |
| --- | --- | --- |
| (a) Données étiquetées ? | **Non** | Pas d'historique de questions/réponses validées sur ce corpus. |
| (b) Vérifiabilité ? | **Exigée** | Chaque réponse doit citer la source exacte (fichier + page). |
| (c) Coût par requête ? | **Modéré** | Un appel LLM + une recherche vectorielle par question. |
| (d) Conséquence d'une erreur ? | **Modérée** | Une réponse fausse est détectable via la source citée. |

**Approche retenue : RAG** (Retrieval-Augmented Generation).

Justification :
- Pas de règles métier car la logique n'est ni connue ni stable (a).
- Pas de ML supervisé car il n'y a pas d'étiquettes (a).
- RAG car la réponse doit être **tirée de documents et citée** (b).
- Le génératif seul hallucinerait : il ne connaît pas nos documents.

---

## 2. Architecture technique

### Vue d'ensemble

    Question → Embedding → Recherche vectorielle (top-k)
             → Contexte fourni au LLM → Réponse + citations

Trois étapes :

1. **Retrieval** — la question est encodée en vecteur, puis comparée
   aux vecteurs des passages indexés dans ChromaDB.
2. **Augmentation** — les k passages les plus proches sont insérés
   dans un prompt qui impose au LLM de ne répondre **qu'à partir
   d'eux**.
3. **Generation** — le LLM (Groq, `openai/gpt-oss-120b`) rédige une
   réponse en citant les sources.

### Structure du dépôt

    rag-agent/
    ├── data/pdfs/               corpus (non versionné)
    ├── src/
    │   ├── config.py            configuration centrale
    │   ├── ingest.py            ingestion : PDF → passages → vecteurs
    │   ├── retrieve.py          recherche vectorielle
    │   └── generate.py          génération sourcée
    ├── rag/
    │   ├── chunking.py          chargement PDF + découpage
    │   ├── embeddings.py        modèle d'embedding local
    │   ├── vector_store.py      ChromaDB
    │   └── llm.py               wrapper Groq
    ├── ui/
    │   ├── app.py               interface Streamlit
    │   └── helpers.py           logique testable
    ├── tests/                   13 tests pytest
    ├── models/chroma/           index vectoriel (non versionné)
    ├── docs/                    cadrage, évaluation, compte rendu
    └── .github/workflows/       intégration continue

### Les trois artefacts, pas un

Comme aux TP1-TP3, nous distinguons :

| Artefact | Dans Git ? | Raison |
| --- | --- | --- |
| Le code | Oui | C'est le travail |
| L'environnement (`.venv/`) | Non | Reconstructible via `pip install` |
| Le corpus (`data/pdfs/*.pdf`) | Non | Volumineux, fourni à part |
| L'index vectoriel (`models/chroma/`) | Non | Reconstructible via `python src/ingest.py` |
| Les métriques et la doc | Oui | Ce sont les preuves |

---

## 3. Choix du LLM et du RAG

### Pourquoi un LLM alors qu'on a déjà les PDF ?

C'est la question centrale du projet. Un PDF est un **texte figé** : il ne
comprend pas une question et ne peut pas la synthétiser. Deux cas :

- **Sans LLM** : la recherche vectorielle seule renvoie des passages
  bruts. Utile, mais l'utilisateur doit lire, comprendre, synthétiser.
- **Sans PDF** : un LLM seul (ChatGPT, Groq…) ne connaît pas nos documents.
  Il **hallucine** une réponse plausible mais fausse.

Le RAG combine les deux forces :
- Le **PDF apporte la connaissance** (le LLM ne peut pas inventer).
- Le **LLM apporte le langage** (il rédige une réponse lisible).
- Ensemble, ils donnent une réponse **précise, rédigée et sourcée**.

### Pourquoi Groq plutôt qu'un LLM local

Notre machine est un ultrabook i5-8365U (4 cœurs, 1.6 GHz, **sans GPU**).
Un LLM local (Ollama, `llama3.2`) mettrait **30 s à 2 min** par réponse et
ferait chauffer le PC. Groq fait tourner des LLM open source sur des puces
spécialisées (LPU) :

| | Groq (API) | Ollama (local) |
| --- | --- | --- |
| Vitesse | 1-3 s | 15-60 s |
| Téléchargement | Aucun | 400 Mo à 2 Go |
| Chauffe PC | Aucune | Élevée |
| Coût | Gratuit (1 000 req/j) | Gratuit |

**Choix : Groq**, modèle `openai/gpt-oss-120b` (remplaçant officiel de
`llama-3.3-70b-versatile`, retiré en août 2026).

---

## 4. Évaluation et limites

### Résultats observés

**Test 1 — Question dans le corpus**

- Question : « Qu'est-ce que le RAG ? »
- Passage récupéré : `cours_ml.pdf, page 1` (distance cosinus **0.697**)
- Le bon passage est classé **premier** sur les 3 passages de l'index.
- Réponse du LLM : ancrée dans l'extrait, avec citation `[cours_ml.pdf, page 1]`.
- **Verdict** : ✅

**Test 2 — Question hors corpus**

- Question : « Quelle est la capitale du Japon ? »
- Le LLM refuse explicitement de répondre (« les extraits fournis ne
  contiennent pas cette information »).
- **Verdict** : ✅ pas d'hallucination.

**Test 3 — Question ambiguë**

- Question : « Comment fonctionne l'apprentissage ? »
- Le LLM croise plusieurs passages et synthétise.
- **Verdict** : ✅ réponse multi-sources.

### Limites

1. **Découpage naïf** : `RecursiveCharacterTextSplitter` avec 512 tokens
   peut couper au milieu d'une phrase. Un découpage sémantique
   améliorerait la cohérence.
2. **Embedding générique** : `all-MiniLM-L6-v2` est multilingue mais
   généraliste. Un modèle spécialisé (par ex. `bge-m3`) améliorerait
   la précision.
3. **Un seul document** dans le corpus de test (`cours_ml.pdf`).
4. **Pas de reranking** : les 4 passages sont fournis au LLM dans
   l'ordre de distance. Un reranker (CrossEncoder) améliorerait la
   sélection.

---

## 5. Ce qui a échoué

Trois incidents notables, tous documentés :

1. **Modèle Groq retiré (erreur 404)**. Nous avions codé en dur
   `llama-3.3-70b-versatile`, retiré par Groq le 16 août 2026. Erreur :
   `The model does not exist or you do not have access to it`.
   Corrigé en basculant sur `openai/gpt-oss-120b`.

2. **Import `langchain.text_splitter` obsolète**. LangChain 0.2 a éclaté
   ses paquets ; l'import est maintenant dans `langchain_text_splitters`.
   Corrigé en installant `langchain-text-splitters`.

3. **CI rouge pendant plusieurs tentatives**. Les tests échouaient en
   intégration continue car les imports en cascade tiraient `torch` et
   `sentence-transformers` (800 Mo), non installés en CI. Corrigé en
   rendant les imports **tardifs** dans `src/generate.py` et
   `src/retrieve.py` — exactement le principe du TP3 (`ui/helpers.py`).

4. **Push HTTP 408** sur les captures d'écran (comme au TP3). Corrigé
   avec `git config --global http.postBuffer 524288000` et
   `git config --global http.version HTTP/1.1`.

---

## 6. Ce qu'on ferait autrement

1. **Découpage sémantique** plutôt que par caractères : utiliser
   `SemanticChunker` ou un découpage basé sur les titres de section.

2. **Reranker** : après la recherche vectorielle, faire passer les
   10 premiers passages dans un CrossEncoder pour ne garder que les
   3 ou 4 plus pertinents.

3. **Évaluation systématique** : construire un jeu de 20 questions de
   référence, mesurer un MRR (Mean Reciprocal Rank) et un taux de
   refus correct.

4. **Multi-documents** : tester sur un corpus plus large (10-20 PDF)
   pour éprouver la recherche vectorielle à l'échelle.

5. **Cache des réponses** : mettre en cache les questions fréquentes
   pour économiser le quota Groq.

---

## 7. Résultats et bilan

| Indicateur | Valeur |
| --- | --- |
| Passages indexés (corpus de test) | 3 |
| Modèle d'embedding | `all-MiniLM-L6-v2` |
| Taille des passages | 512 tokens, chevauchement 50 |
| Modèle LLM | `openai/gpt-oss-120b` (Groq) |
| Temps de réponse moyen | ~2 secondes |
| Tests pytest (local) | **13 passed** |
| Tests pytest (CI) | **10 passed** en 29 s |
| CI GitHub Actions | **Verte** ✅ |

### Ce que le projet démontre

- Un RAG complet, fonctionnel, avec citations sourcées.
- Une architecture séparée (rag / src / ui / tests) qui facilite les
  tests et l'évolution.
- Une intégration continue qui rejoue les tests à chaque push.
- Une documentation du **pourquoi** (pas seulement du **comment**).

---

## 8. Répartition du travail

| Membre | Contributions |
| --- | --- |
| **Othmane Amaadour** | Structure du dépôt, ingestion (`chunking.py`, `embeddings.py`, `vector_store.py`), configuration Groq, génération sourcée, CI GitHub Actions |
| **Mohcine Errachid** | Interface Streamlit (`ui/app.py`, `ui/helpers.py`), tests pytest, documentation (`evaluation.md`, fiche de cadrage), rédaction du compte rendu |

Le travail a été mené en binôme, avec des points de synchronisation
réguliers sur l'architecture et les choix techniques. Chaque membre a
contribué au dépôt Git (vérifiable avec `git shortlog -sn --all`).

---

*Version remise le 16 octobre 2026, figée par l'étiquette Git
`rendu-final`.*
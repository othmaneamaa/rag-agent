# Cadrage — Agent de recherche documentaire

## Le besoin

Permettre à un utilisateur d'interroger en langage naturel un corpus
hétérogène (PDF, articles, pages internes) et d'obtenir des réponses
**sourcées**, c'est-à-dire accompagnées des documents dont elles sont tirées.

## La grille de décision

Nous appliquons la grille du cours : (a) données étiquetées,
(b) vérifiabilité, (c) coût par requête, (d) conséquence d'une erreur.

| Critère | Réponse | Commentaire |
| --- | --- | --- |
| (a) Données étiquetées ? | **Non** | Il n'existe pas d'historique de questions/réponses validées sur ce corpus. |
| (b) Vérifiabilité ? | **Exigée** | Chaque réponse doit citer la source exacte (fichier + page). |
| (c) Coût par requête ? | Modéré | Un appel LLM + une recherche vectorielle par question. |
| (d) Conséquence d'une erreur ? | Modérée | Une réponse fausse est détectable par l'utilisateur en consultant la source citée. |

## L'approche retenue

- [ ] Règles métier
- [ ] Machine learning
- [x] **RAG** (Retrieval-Augmented Generation)
- [ ] Génératif seul

## Justification

La logique n'est ni connue ni stable (a) → pas de règles métier.
Il n'y a pas d'étiquettes disponibles (a) → pas de ML supervisé.
La réponse doit être tirée de documents et **citée** (b) → c'est exactement
le cas d'usage du RAG.
Le génératif seul hallucinerait, car il ne connaît pas nos documents.

## Architecture

    Question → Embedding → Recherche vectorielle (top-k)
             → Contexte fourni au LLM → Réponse + citations

Trois étapes : **retrieval** (ChromaDB + sentence-transformers),
**augmentation** (assemblage du prompt avec les passages),
**generation** (LLM Groq, modèle `openai/gpt-oss-120b`).

## Critères de succès

1. Une question sur le corpus obtient une réponse **ancrée** dans les passages.
2. Une question **hors corpus** est refusée (pas d'invention).
3. Chaque réponse cite **au moins une source** au format `[fichier, page]`.
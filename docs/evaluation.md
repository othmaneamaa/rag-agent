# Évaluation du RAG

## Ce qu'on mesure

Le cadre CLEAR-RAG propose cinq dimensions. Nous en retenons trois,
mesurables sur notre projet étudiant.

| Dimension | Métrique | Comment on la mesure |
| --- | --- | --- |
| **Retrieval Quality** | Distance cosinus du bon passage | Distance du 1er résultat pour une question dont on connaît la source |
| **Answer Faithfulness** | Taux de refus hors corpus | Le LLM refuse-t-il de répondre quand l'info n'est pas dans les passages ? |
| **Citation Quality** | Présence de la source | Chaque réponse cite-t-elle au moins une source ? |

### Nettoyage du corpus

Avant l'indexation, `rag/chunking.py` ignore les pages non informatives (garde, remerciements, liens LinkedIn, bibliographie). Sur les 158 pages des 5 PDF, **19 pages** ont été ignorées, laissant **139 pages** et **508 passages** indexés (au lieu de 532).

### Seuil de similarité

`src/retrieve.py` filtre les passages dont la distance cosinus dépasse **0.55**. Ce seuil a été calibré sur le corpus :
- 0.259 (régularisation), 0.394 (Spider-Man), 0.516 (Streamlit) → passages pertinents, conservés ;
- 0.542 à 0.611 (bruit, malédiction de la profondeur) → filtrés ;
- > 0.55 pour une question hors corpus (capitale du Japon) → **aucun passage**, donc refus déterministe sans appel LLM.

Ce seuil évite d'envoyer au LLM des passages hors sujet, et rend le refus hors corpus **déterministe** (ce n'est plus le LLM qui décide, c'est le code).

### Test 1 — Question sur un document spécifique

- **Question** : « Qu'est-ce que la régularisation ? »
- **Passages récupérés** : 4 passages de `cours_deep_learning.pdf` (distances 0.259 à 0.527).
- **Verdict** : ✅ distance très basse (0.259), le bon document et la bonne page.

### Test 2 — Question sur un autre document

- **Question** : « Comment fonctionne Streamlit ? »
- **Passages récupérés** : 4 passages de `guide_streamlit.pdf` (distances 0.516 à 0.559).
- **Verdict** : ✅ le bon document, avec la définition de Streamlit en passage 2.

### Test 3 — Question sur un troisième document

- **Question** : « Qui est Spider-Man ? »
- **Passages récupérés** : 2 passages de `Marvel.pdf` (distances 0.394 et 0.478), puis 1 passage de `cours_deep_learning.pdf` (0.542).
- **Verdict** : ✅ le bon document est classé premier, avec une distance très basse.

### Test 4 — Question hors corpus

- **Question** : « Qu'est-ce que le RAG ? »
- **Résultat** : le LLM refuse explicitement de répondre (« La réponse ne se trouve pas dans les extraits fournis »), en citant les extraits consultés.
- **Verdict** : ✅ pas d'hallucination, même avec 4 passages hors sujet en contexte.

### Évaluation quantitative (20 questions)

Nous avons construit un jeu de 20 questions (`data/questions_reference.csv`) : 15 dans le corpus, 5 hors corpus. Le script `src/evaluate_rag.py` mesure deux métriques :
- **Context Precision@3** : le bon document est-il dans le top 3 ?
- **Taux de refus** : les questions hors corpus renvoient-elles 0 passage ?

| Seuil | Context Precision@3 | Taux de refus |
|-------|---------------------|---------------|
| 0.50 | 9/15 = 60,0 % | 5/5 = 100 % |
| **0.55** | **11/15 = 73,3 %** | 3/5 = 60 % |

**Choix : 0.55.** Il préserve la précision (73,3 %), au prix de 2 questions hors corpus qui passent le seuil et sont refusées par le LLM (comportement à deux niveaux : le seuil filtre le plus évident, le LLM filtre le reste). À 0.50, le refus est parfait (100 %), mais 2 bons passages sont perdus (précision 60 %).

### Conclusion

Le retrieval distingue **3 documents différents** (`cours_deep_learning.pdf`, `guide_streamlit.pdf`, `Marvel.pdf`) selon la question, et le LLM **refuse** de répondre quand l'information n'est pas dans le corpus. Le nettoyage des pages non informatives a supprimé 19 pages de bruit, et le seuil de similarité à 0.55 rend le refus hors corpus déterministe pour les cas les plus évidents.

## Limites observées

1. **Découpage naïf** : `RecursiveCharacterTextSplitter` avec 512 tokens peut couper au milieu d'une phrase. Un découpage sémantique améliorerait la cohérence.
2. **Embedding générique** : `all-MiniLM-L6-v2` est multilingue mais généraliste. Un modèle spécialisé (par ex. `bge-m3`) améliorerait la précision.
3. **Seuil de similarité fixe** : 0.55 fonctionne bien sur ce corpus, mais un seuil adaptatif (selon la longueur de la question ou le nombre de documents) serait plus robuste. Les 2 questions hors corpus qui passent le seuil sont refusées par le LLM, pas par le seuil.
4. **Pas de reranking** : les 4 passages récupérés sont fournis au LLM dans l'ordre de distance. Un reranker (CrossEncoder) améliorerait la sélection.
5. **Corpus sans définition du RAG** : la question « Qu'est-ce que le RAG ? » est hors corpus, ce qui est un bon test de refus, mais empêche de valider la définition.

## Ce qu'on ferait avec plus de temps

- Mesurer un **MRR** sur 20 questions de référence
- Tester plusieurs modèles d'embedding
- Ajouter un **reranker** (par ex. `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- Comparer RAG vs LLM seul sur les mêmes questions
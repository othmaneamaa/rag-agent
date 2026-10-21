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
- **Passages récupérés** : 2 passages de `Marvel.pdf` (distances 0.394 et 0.478), puis 2 passages d'autres documents (bruit).
- **Verdict** : ✅ le bon document est classé premier, avec une distance très basse.

### Test 4 — Question hors corpus

- **Question** : « Qu'est-ce que le RAG ? »
- **Résultat** : le LLM refuse explicitement de répondre (« La réponse ne se trouve pas dans les extraits fournis »), en citant les extraits consultés.
- **Verdict** : ✅ pas d'hallucination, même avec 4 passages hors sujet en contexte.

### Conclusion

Le retrieval distingue **3 documents différents** (`cours_deep_learning.pdf`, `guide_streamlit.pdf`, `Marvel.pdf`) selon la question, et le LLM **refuse** de répondre quand l'information n'est pas dans le corpus. Le nettoyage des pages non informatives a supprimé 19 pages de bruit.

## Limites observées

1. **Découpage naïf** : `RecursiveCharacterTextSplitter` avec 512 tokens peut couper au milieu d'une phrase. Un découpage sémantique améliorerait la cohérence.
2. **Embedding générique** : `all-MiniLM-L6-v2` est multilingue mais généraliste. Un modèle spécialisé (par ex. `bge-m3`) améliorerait la précision.
3. **Pas de seuil de distance** : même les passages à 0.6 (médiocres) sont renvoyés au LLM. Un seuil de 0.55 éliminerait le bruit (par exemple, les passages 3 et 4 de « Qui est Spider-Man ? »).
4. **Pas de reranking** : les 4 passages récupérés sont fournis au LLM dans l'ordre de distance. Un reranker (CrossEncoder) améliorerait la sélection.
5. **Corpus sans définition du RAG** : la question « Qu'est-ce que le RAG ? » est hors corpus, ce qui est un bon test de refus, mais empêche de valider la définition.

## Ce qu'on ferait avec plus de temps

- Mesurer un **MRR** sur 20 questions de référence
- Tester plusieurs modèles d'embedding
- Ajouter un **reranker** (par ex. `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- Comparer RAG vs LLM seul sur les mêmes questions
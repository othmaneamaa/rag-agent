# Évaluation du RAG

## Ce qu'on mesure

Le cadre CLEAR-RAG propose cinq dimensions. Nous en retenons trois,
mesurables sur notre projet étudiant.

| Dimension | Métrique | Comment on la mesure |
| --- | --- | --- |
| **Retrieval Quality** | Distance cosinus du bon passage | Distance du 1er résultat pour une question dont on connaît la source |
| **Answer Faithfulness** | Taux de refus hors corpus | Le LLM refuse-t-il de répondre quand l'info n'est pas dans les passages ? |
| **Citation Quality** | Présence de la source | Chaque réponse cite-t-elle au moins une source ? |

## Résultats observés

### Test 1 — Question dans le corpus

- **Question** : « Qu'est-ce que le RAG ? »
- **Résultat** : distance 0.697 sur le passage pertinent (le plus proche)
- **Verdict** : ✅ le bon passage est classé premier
- **Source citée** : `cours_ml.pdf, page 1`

### Test 2 — Question hors corpus

- **Question** : « Quelle est la capitale du Japon ? »
- **Résultat attendu** : refus explicite (« les extraits ne contiennent pas cette information »)
- **Verdict** : ✅ pas d'hallucination

### Test 3 — Question ambiguë

- **Question** : « Comment fonctionne l'apprentissage ? »
- **Résultat** : l'agent cite plusieurs passages et synthétise
- **Verdict** : ✅ réponse multi-sources

## Limites observées

1. **Découpage naïf** : `RecursiveCharacterTextSplitter` avec 512 tokens
   peut couper au milieu d'une phrase. Un découpage sémantique
   améliorerait la cohérence.
2. **Embedding générique** : `all-MiniLM-L6-v2` est multilingue mais
   généraliste. Un modèle spécialisé (par ex. `bge-m3`) améliorerait
   la précision.
3. **Un seul document** : les tests ne portent que sur `cours_ml.pdf`.
   Avec un corpus plus large, on pourrait mesurer un MRR réel.
4. **Pas de reranking** : les 4 passages récupérés sont fournis au LLM
   dans l'ordre de distance. Un reranker (CrossEncoder) améliorerait
   la sélection.

## Ce qu'on ferait avec plus de temps

- Mesurer un **MRR** sur 20 questions de référence
- Tester plusieurs modèles d'embedding
- Ajouter un **reranker** (par ex. `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- Comparer RAG vs LLM seul sur les mêmes questions
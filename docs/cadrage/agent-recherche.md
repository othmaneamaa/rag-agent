# Cadrage : agent de recherche documentaire

## Le besoin

Permettre à un utilisateur d'interroger en langage naturel un corpus de documents en français, et d'obtenir des réponses **sourcées**, c'est-à-dire accompagnées des passages exacts dont elles sont tirées, ou un **refus** quand le corpus ne contient pas la réponse.

## Le corpus

[Piaf (French Questions & Answers)](https://www.kaggle.com/datasets/thedevastator/unlock-the-power-of-reading-comprehension-with-p), publié sur Kaggle sous licence CC0, issu du projet PIAF d'Etalab :

- **761 paragraphes** de Wikipédia en français, de 500 à 996 caractères, sur **191 sujets** (histoire, géographie, sport, sciences, culture) ;
- **3 835 questions** écrites par des humains, chacune sur un paragraphe précis, avec sa réponse.

Il remplace les 5 PDF de la version 0.1.0, qui n'étaient pas dans le dépôt : personne d'autre que nous ne pouvait reconstruire l'index ni rejouer l'évaluation. PIAF se télécharge en une commande (`python src/telecharger.py`), et un extrait de 10 sujets est versionné (`data/corpus_exemple.csv`) pour les tests et la CI.

## La grille de décision

Nous appliquons la grille du cours : (a) données étiquetées, (b) vérifiabilité, (c) coût par requête, (d) conséquence d'une erreur.

| Critère | Réponse | Commentaire |
| --- | --- | --- |
| (a) Données étiquetées ? | **Pour mesurer, pas pour apprendre** | Chaque question connaît son paragraphe : de quoi évaluer la recherche, pas de quoi entraîner un modèle qui répondrait à toute question. |
| (b) Vérifiabilité ? | **Exigée** | Chaque réponse cite ses paragraphes au format `[titre, identifiant]`. |
| (c) Coût par requête ? | Modéré | Une recherche vectorielle locale, puis un appel à Groq ; aucun appel quand l'agent refuse au seuil. |
| (d) Conséquence d'une erreur ? | Modérée | Une réponse fausse se détecte en ouvrant le passage cité. Une invention sans source serait plus grave : d'où les deux refus. |

## L'approche retenue

- [ ] Règles métier
- [ ] Machine learning
- [x] **RAG** (Retrieval-Augmented Generation)
- [ ] Génératif seul

## Justification

Les questions sont ouvertes et la logique n'est pas écrite d'avance (a) : pas de règles métier. Les étiquettes de PIAF mesurent la recherche mais ne suffisent pas à apprendre à répondre à n'importe quelle question (a) : pas de ML supervisé seul. La réponse doit être tirée du corpus et citée (b) : c'est le cas d'usage du RAG. Un modèle de langage seul inventerait des réponses plausibles, sans source vérifiable (d).

## Architecture

    Question → embedding (multilingual-e5-small) → ChromaDB, top 4, distance ≤ 0,19
             → aucun passage : refus, sans appel au LLM
             → sinon : prompt borné aux extraits → Groq (gpt-oss-120b)
                       → « HORS_CORPUS » : refus ; sinon réponse + citations

## Critères de succès

1. **Recherche** : le bon paragraphe est dans les 3 premiers pour au moins 80 % des questions de PIAF (mesuré : 83,2 %).
2. **Refus** : une question hors corpus n'obtient pas de réponse inventée ; au seuil, au moins la moitié est refusée sans appel au LLM (mesuré : 66,7 % sur 30 questions), le reste par le marqueur `HORS_CORPUS`.
3. **Citations** : chaque réponse cite au moins un passage du corpus, avec son identifiant.

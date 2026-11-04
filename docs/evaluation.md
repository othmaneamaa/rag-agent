# Évaluation de l'agent

> Toutes les valeurs viennent de nos exécutions : `python src/evaluate.py` et `python src/courbe_seuil.py` sur le corpus PIAF complet, avec les versions épinglées de `requirements.txt`.

## Ce qu'on mesure, et pourquoi c'est possible sans annoter

Chaque question de PIAF a été écrite par une personne qui lisait un paragraphe précis. On sait donc, pour les 3 835 questions, quel est le bon paragraphe : la recherche se mesure automatiquement, comme le F1 de sentiment-app se mesure sur son jeu de test.

| Dimension | Métrique | Lecture |
| --- | --- | --- |
| Qualité de la recherche | rappel@1, rappel@3 | Part des questions dont le bon paragraphe est 1er, ou dans les 3 premiers |
| | MRR@10 | Moyenne de 1/rang du bon paragraphe (1 s'il est 1er, 0,5 s'il est 2e, 0 s'il n'est pas dans les 10) |
| Refus | Refus hors corpus | Part des 30 questions de `data/questions_hors_corpus.csv` sans aucun passage sous le seuil |
| | Réponses | Part des questions du corpus qui gardent au moins un passage sous le seuil |

## Le choix du modèle d'embedding

Même corpus, mêmes 3 835 questions, recherche exacte :

| Modèle | rappel@1 | rappel@3 | MRR@10 |
| --- | --- | --- | --- |
| `all-MiniLM-L6-v2` (version 0.1.0) | 0,340 | 0,481 | 0,423 |
| TF-IDF, mots et paires de mots (référence sans réseau de neurones) | 0,613 | 0,763 | 0,697 |
| `paraphrase-multilingual-MiniLM-L12-v2` | 0,448 | 0,611 | 0,546 |
| **`intfloat/multilingual-e5-small`** | **0,668** | **0,832** | **0,757** |

Le modèle de la version 0.1.0 a été entraîné surtout sur de l'anglais : sur des questions en français, il fait moins bien qu'une simple recherche par mots-clés. `multilingual-e5-small` est le seul à battre nettement TF-IDF. Il demande un préfixe `query: ` devant les questions et `passage: ` devant les paragraphes ; `rag_agent/index.py` les ajoute.

**Seuil d'acceptation** : rappel@3 ≥ 0,80 (`SEUIL_RAPPEL_3` dans `src/evaluate.py`). Atteint : 0,832.

### Un défaut trouvé en route : la recherche approchée

Avec les réglages par défaut de ChromaDB 0.5 (recherche HNSW, `ef` = 10), le rappel@3 tombait à 0,803, et deux évaluations successives ne donnaient pas les mêmes chiffres : la recherche approchée manquait des voisins. Avec `hnsw:search_ef` = 200, sur 761 passages, la recherche redevient exacte et reproductible, et retrouve les chiffres du calcul exact (0,832).

## Le seuil de distance

`python src/courbe_seuil.py`, top 4 :

| Seuil | Répond | Bon passage gardé | Refus hors corpus |
| --- | --- | --- | --- |
| 0,16 | 60,6 % | 54,1 % | 100,0 % |
| 0,17 | 73,6 % | 64,8 % | 93,3 % |
| 0,18 | 85,9 % | 74,2 % | 76,7 % |
| **0,19** | **95,1 %** | **80,9 %** | **66,7 %** |
| 0,20 | 98,6 % | 84,2 % | 36,7 % |
| 0,22 | 100,0 % | 85,4 % | 3,3 % |

Les distances se chevauchent : pour 90 % des questions du corpus, le passage le plus proche est entre 0,106 et 0,190 (5e et 95e centiles) ; pour les 30 questions hors corpus, entre 0,166 et 0,226. Aucun seuil ne sépare tout.

**Choix : 0,19.** L'agent répond encore à 95 % des questions légitimes, et refuse deux tiers des questions hors corpus sans même appeler le modèle de langage. À 0,17, le refus hors corpus serait presque parfait (93 %), mais l'agent ne répondrait plus qu'à 74 % des questions légitimes. Les questions hors corpus qui passent le seuil rencontrent le second garde-fou : le prompt impose au modèle de répondre `HORS_CORPUS` si les extraits ne contiennent pas la réponse, et le code transforme ce mot en refus.

Les 10 questions hors corpus qui passent le seuil, avec le sujet le plus proche :

| Distance | Question | Sujet le plus proche |
| --- | --- | --- |
| 0,166 | Quelle est la meilleure crème solaire pour les enfants ? | Bronzage |
| 0,166 | Combien de joueurs compte une équipe de rugby ? | Jeux olympiques d'été de 2016 |
| 0,172 | Quels sont les symptômes du diabète ? | Syndrome d'apnées du sommeil |
| 0,176 | Qu'est-ce que la photosynthèse ? | Mycélium |
| 0,177 | Quelle est la différence entre un virus informatique et un ver ? | Virus coxsackie B |
| 0,179 | Quelle est la capitale de l'Australie ? | Rome |
| 0,179 | Quelle est la recette du couscous ? | Hongrie |
| 0,181 | Combien de temps faut-il cuire des pâtes al dente ? | France |
| 0,184 | Comment calculer un pourcentage ? | Couleur du Web |
| 0,189 | Comment entretenir un aquarium d'eau douce ? | Événement Azolla |

Ce sont des voisinages de thème (bronzage et crème solaire, virus et virus informatique), pas des réponses : c'est exactement le cas que le second garde-fou doit traiter.

## Limites mesurées

1. **Les questions qui ne se comprennent qu'avec leur paragraphe.** 349 questions (9,1 %) ne retrouvent pas leur paragraphe dans les 10 premiers. Beaucoup ont été écrites en lisant le texte : « De quelle organisation fait-elle partie ? », « qui dédouble la représentation ». Sans le paragraphe, personne ne pourrait y répondre ; c'est une limite du jeu autant que du modèle.
2. **Les questions courtes.** Rappel@3 de 0,763 pour les questions de 8 mots ou moins (1 522 questions), contre 0,877 au-delà (2 313 questions).
3. **Les sujets aux paragraphes voisins.** Les batailles et les sièges sont les moins bien servis : Siège de Gibraltar 0,44, Bataille de Savenay 0,48, Guerre d'indépendance espagnole 0,52 (25 questions chacun). Leurs paragraphes partagent le même vocabulaire militaire. À l'opposé, Best-seller, Bourse de Brême et Calendrier hégirien sont à 1,00.
4. **Le refus est mesuré sur 30 questions seulement**, écrites par nous : 66,7 % est un ordre de grandeur, pas une mesure précise.
5. **La rédaction n'est pas mesurée automatiquement.** La fidélité des réponses de Groq aux extraits et la justesse des citations se vérifient à la main ; les tests vérifient seulement que le prompt borne le modèle aux extraits et que le marqueur de refus est respecté.

## Ce qu'on ferait avec plus de temps

- Un modèle plus grand (`multilingual-e5-base`, `bge-m3`) et un reranker (cross-encoder) sur les 10 premiers passages, mesurés avec le même script.
- Une recherche hybride, TF-IDF et embeddings : TF-IDF atteint déjà 0,763 au rappel@3, et les mots exacts (noms propres, dates) pourraient compléter les embeddings. À mesurer avec le même script.
- Un jeu de questions hors corpus plus grand, et une mesure de la fidélité des réponses générées.

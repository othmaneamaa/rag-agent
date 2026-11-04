# Fiche modèle : rag-agent

> Sur le modèle de la fiche de sentiment-app : ce que fait l'agent, sur quelles données, ce qu'il vaut, où il se trompe, et dans quelles conditions s'en servir. Les tests (`tests/test_fiche_modele.py`) vérifient que les chiffres cités sont ceux de `models/metrics.json`.

## Identité du modèle

Agent de recherche documentaire `rag-agent`, version 0.2.0 du paquet `rag_agent`. Recherche : embeddings `intfloat/multilingual-e5-small` (sentence-transformers 3.2), index ChromaDB 0.5 en distance cosinus, 4 passages par question. Rédaction : `openai/gpt-oss-120b`, servi par Groq. L'index est construit par `python src/ingest.py` et n'est ni dans Git ni dans la roue ; il retient le nom du modèle d'embedding qui l'a produit.

## Usage prévu

Répondre à des questions factuelles en français sur les 191 sujets du corpus PIAF, avec des citations vérifiables, dans le cadre d'un projet de cours. L'utilisateur lit la réponse, puis ouvre les passages cités pour vérifier. La recherche seule (`/search`, `rag search`) sert aussi à explorer le corpus sans clé Groq.

## Usages exclus

- Répondre sur des sujets hors des 191 articles du corpus : l'agent doit refuser, et ne sait rien d'autre.
- Servir de source unique pour une décision médicale, juridique ou financière : Wikipédia n'est pas une source d'autorité, et l'extrait peut être daté.
- Indexer des documents personnels ou confidentiels sans revoir la configuration : les questions et les extraits partent chez Groq, un service externe.

## Données d'entraînement

Aucun modèle n'est entraîné ici : les modèles sont pré-entraînés, et l'agent indexe le corpus. Corpus : jeu Piaf (French Questions & Answers) de Kaggle, licence CC0, 761 paragraphes de Wikipédia en français (500 à 996 caractères) sur 191 sujets, et 3 835 questions écrites par des humains, qui servent uniquement à l'évaluation. Le corpus ne couvre ni l'actualité, ni les sujets techniques courants (informatique, cuisine, santé au quotidien), ni d'autres langues.

## Performances mesurées

Sur les 3 835 questions de PIAF (`models/metrics.json`) : rappel@1 0.668, rappel@3 0.832, MRR@10 0.757. Le seuil d'acceptation, rappel@3 ≥ 0,80, est atteint. Au seuil de distance de 0.19, l'agent répond à 95,1 % des questions du corpus, avec le bon passage parmi ceux gardés dans 80,9 % des cas, et refuse sans appeler le LLM 66,7 % des 30 questions hors corpus. Le modèle de la version 0.1.0, `all-MiniLM-L6-v2`, n'atteignait que 0,481 au rappel@3.

## Limites connues

9,1 % des questions ne retrouvent pas leur paragraphe dans les 10 premiers ; beaucoup ne se comprennent qu'avec lui (« De quelle organisation fait-elle partie ? »). Les questions de 8 mots ou moins sont moins bien servies (rappel@3 0,763 contre 0,877). Un tiers des questions hors corpus passent le seuil, par voisinage de thème : « Quelle est la meilleure crème solaire pour les enfants ? » tombe sur l'article Bronzage, à 0,166 ; c'est alors au LLM de répondre `HORS_CORPUS`, ce que les tests vérifient avec un faux modèle mais que seule une relecture vérifie avec Groq.

## Biais mesurés

Les sujets dont les paragraphes partagent le même vocabulaire sont moins bien servis : Siège de Gibraltar (rappel@3 0,44), Bataille de Savenay (0,48), Guerre d'indépendance espagnole (0,52), sur 25 questions chacun, contre 1,00 pour Best-seller ou Calendrier hégirien. Le corpus reflète aussi les choix de Wikipédia et des annotateurs de PIAF : beaucoup d'histoire militaire et de France, peu de sujets du quotidien. Les effectifs par sujet sont petits : ces écarts sont des signaux, pas des mesures précises.

## Explicabilité

Chaque réponse s'explique par ses passages : l'API renvoie les extraits transmis au modèle, leur identifiant et leur distance à la question, et la réponse cite ceux qu'elle utilise. Un refus s'explique aussi : soit aucun passage n'était sous le seuil (la page « Chercher des passages » le montre), soit le modèle a jugé que les extraits ne contenaient pas la réponse.

## Conditions d'usage

Un humain vérifie toute réponse avant de s'en servir, en ouvrant au moins un passage cité ; une réponse sans citation, ou dont la citation ne contient pas l'information, est à écarter. Un refus est un résultat normal, pas une panne. On ne décide jamais rien d'important sur la seule foi de l'agent, et on ne lui soumet pas de données personnelles, puisque les extraits partent chez Groq. La clé Groq reste dans `.env`, hors de Git.

## Maintenance

Réévaluer (`python src/evaluate.py`, puis `python src/courbe_seuil.py`) à chaque changement de modèle d'embedding, de corpus ou de seuil, et mettre cette fiche à jour : `tests/test_metriques.py` échoue si `models/metrics.json` ne correspond plus aux réglages. Changer de modèle d'embedding impose de reconstruire l'index : `rag_agent.index.ouvrir()` refuse un index construit avec un autre modèle. Le paquet trouve l'index par la variable `RAG_INDEX`. Groq retire des modèles (`llama-3.3-70b-versatile` l'a été en 2026) : le modèle se change par la variable `LLM_MODEL`.

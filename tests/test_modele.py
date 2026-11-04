"""Le vrai modèle d'embedding, sur le corpus d'exemple versionné (81 paragraphes, 417 questions).

Ces tests téléchargent le modèle (environ 470 Mo) au premier lancement ; la CI les exécute.
Ils vérifient ce que l'encodeur de hachage ne peut pas : que la recherche marche sur du vrai
français, et que les préfixes E5 sont bien appliqués.
"""
from pathlib import Path

import pytest

pytest.importorskip("sentence_transformers")
pytestmark = pytest.mark.modele

from rag_agent import corpus, evaluation  # noqa: E402
from rag_agent.index import construire, encodeur_par_defaut  # noqa: E402

RACINE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def index_exemple(tmp_path_factory):
    df = corpus.lire(RACINE / "data" / "corpus_exemple.csv")
    index = construire(corpus.passages(df), tmp_path_factory.mktemp("chroma"), encodeur_par_defaut())
    return index, corpus.questions(df)


def test_la_recherche_retrouve_le_bon_paragraphe(index_exemple):
    index, questions = index_exemple
    trouves = index.chercher_lot(questions["question"].tolist(), k=3, seuil=None)
    rangs = [
        evaluation.rang(attendu, [p["passage_id"] for p in liste])
        for attendu, liste in zip(questions["passage_id"], trouves, strict=True)
    ]
    assert evaluation.rappel_a(3, rangs) >= 0.85, "sur 10 sujets seulement, le rappel@3 doit être élevé"


def test_une_question_precise(index_exemple):
    index, _ = index_exemple
    premier = index.chercher("Quand Antoine Lavoisier est-il né ?", k=1, seuil=None)[0]
    assert premier["titre"] == "Antoine Lavoisier"
    assert "1743" in premier["texte"]


def test_une_question_hors_corpus_est_refusee_au_seuil(index_exemple):
    from rag_agent import config

    index, _ = index_exemple
    assert index.chercher("Comment faire une pâte à crêpes ?", seuil=config.SEUIL_DISTANCE) == []


def test_les_prefixes_e5_sont_appliques():
    encodeur = encodeur_par_defaut()
    assert encodeur._prefixe_question == "query: " and encodeur._prefixe_passage == "passage: "

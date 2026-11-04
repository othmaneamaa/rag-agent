"""Le corpus PIAF : lecture du format Kaggle, nettoyage, passages, questions."""
from pathlib import Path

import pandas as pd
import pytest

from rag_agent import corpus

RACINE = Path(__file__).resolve().parents[1]
EXEMPLE = RACINE / "data" / "corpus_exemple.csv"
HORS_CORPUS = RACINE / "data" / "questions_hors_corpus.csv"


@pytest.mark.parametrize(
    ("brut", "propre"),
    [
        ("<i>Vizzavona</i> (navire)", "Vizzavona (navire)"),
        ('<i><span class="lang-la" lang="la">Pomerium</span></i>', "Pomerium"),
        ('Ferdinand <abbr class="abbr" title="premier">I<sup>er</sup></abbr>', "Ferdinand Ier"),
        ("Moscou", "Moscou"),
        ("Pirates &amp; corsaires", "Pirates & corsaires"),
    ],
)
def test_les_titres_perdent_leur_balisage(brut, propre):
    assert corpus.nettoyer_titre(brut) == propre


def test_l_identifiant_est_stable_et_court():
    texte = "Moscou est la capitale de la Russie."
    assert corpus.identifiant(texte) == corpus.identifiant(texte)
    assert corpus.identifiant(texte) != corpus.identifiant(texte + " ")
    assert len(corpus.identifiant(texte)) == 9 and corpus.identifiant(texte).startswith("P")


def test_un_fichier_sans_les_colonnes_de_piaf_est_refuse(tmp_path):
    chemin = tmp_path / "autre.csv"
    pd.DataFrame({"text": ["bonjour"]}).to_csv(chemin, index=False)
    with pytest.raises(ValueError, match="colonnes absentes"):
        corpus.lire(chemin)


def test_un_passage_par_paragraphe(tmp_path):
    chemin = tmp_path / "mini.csv"
    pd.DataFrame({
        "id": ["1", "2", "3"], "title": ["A", "A", "B"],
        "context": ["Texte un.", "Texte un.", "Texte deux."],
        "question": ["Q1 ?", "Q2 ?", "Q3 ?"], "answers": ["{}", "{}", "{}"],
    }).to_csv(chemin, index=False)
    df = corpus.lire(chemin)
    assert [p.texte for p in corpus.passages(df)] == ["Texte un.", "Texte deux."]
    questions = corpus.questions(df)
    assert len(questions) == 3
    assert questions["passage_id"].nunique() == 2


def test_le_corpus_d_exemple_est_versionne_et_complet():
    """L'extrait de PIAF sur lequel tournent la CI et les tests du vrai modèle."""
    assert EXEMPLE.exists(), "data/corpus_exemple.csv absent : python src/corpus_exemple.py"
    df = corpus.lire(EXEMPLE)
    assert df["titre"].nunique() == 10
    assert len(corpus.passages(df)) == 81
    assert len(df) == 417


def test_les_questions_hors_corpus_sont_trente_et_uniques():
    questions = pd.read_csv(HORS_CORPUS)["question"]
    assert len(questions) == 30
    assert questions.is_unique
    assert all(q.strip().endswith("?") for q in questions)

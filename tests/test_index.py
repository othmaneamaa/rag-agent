"""L'index vectoriel, avec l'encodeur de hachage : construire, chercher, seuil, ouvrir."""
import pytest
from conftest import PASSAGES, EncodeurHachage

from rag_agent.index import construire, ouvrir


def test_l_index_contient_un_passage_par_paragraphe(index):
    assert index.taille == len(PASSAGES)
    assert index.modele == "hachage-test"


def test_reconstruire_remplace_au_lieu_de_doubler(tmp_path, encodeur):
    """L'ancien ingest.py ajoutait les passages à chaque lancement : l'index doublait."""
    construire(PASSAGES, tmp_path, encodeur)
    index = construire(PASSAGES[:2], tmp_path, encodeur)
    assert index.taille == 2


def test_la_question_trouve_son_passage_en_premier(index):
    resultats = index.chercher("Où est né le chimiste Lavoisier ?", k=4, seuil=None)
    assert resultats[0]["titre"] == "Antoine Lavoisier"
    distances = [r["distance"] for r in resultats]
    assert distances == sorted(distances)
    assert set(resultats[0]) == {"passage_id", "titre", "texte", "distance"}


def test_le_seuil_ecarte_les_passages_eloignes(index):
    assert index.chercher("recette de la pâte à crêpes", k=4, seuil=0.5) == []
    assert all(r["distance"] <= 0.9 for r in index.chercher("capitale", k=4, seuil=0.9))


def test_chercher_lot_donne_la_meme_chose_que_chercher(index):
    questions = ["capitale de la Russie", "tortue des Galápagos"]
    assert index.chercher_lot(questions, k=2, seuil=None) == [
        index.chercher(q, k=2, seuil=None) for q in questions
    ]


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_une_question_vide_est_refusee(index, question):
    with pytest.raises(ValueError, match="question vide"):
        index.chercher(question)


def test_ouvrir_un_index_absent_explique_le_remede(tmp_path):
    with pytest.raises(FileNotFoundError, match="python src/ingest.py.*RAG_INDEX"):
        ouvrir(tmp_path / "rien", EncodeurHachage())


def test_ouvrir_un_dossier_sans_collection(tmp_path):
    (tmp_path / "vide").mkdir()
    with pytest.raises(FileNotFoundError, match="index vide"):
        ouvrir(tmp_path / "vide", EncodeurHachage())


def test_ouvrir_avec_un_autre_modele_est_refuse(tmp_path, encodeur):
    construire(PASSAGES, tmp_path, encodeur)

    class Autre(EncodeurHachage):
        nom = "un-autre-modele"

    with pytest.raises(ValueError, match="construit avec hachage-test"):
        ouvrir(tmp_path, Autre())


def test_rag_index_indique_ou_chercher(monkeypatch, tmp_path, encodeur):
    from rag_agent import config

    construire(PASSAGES, tmp_path / "ailleurs", encodeur)
    monkeypatch.setenv("RAG_INDEX", str(tmp_path / "ailleurs"))
    assert config.chemin_index() == tmp_path / "ailleurs"
    assert ouvrir(encodeur=encodeur).taille == len(PASSAGES)

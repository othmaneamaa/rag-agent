"""Les mesures de la recherche, sur des valeurs calculées à la main."""
import pytest

from rag_agent import evaluation


def test_le_rang_du_passage_attendu():
    assert evaluation.rang("b", ["a", "b", "c"]) == 2
    assert evaluation.rang("z", ["a", "b"]) == 0


def test_rappel_et_mrr():
    rangs = [1, 2, 0, 4]
    assert evaluation.rappel_a(1, rangs) == 0.25
    assert evaluation.rappel_a(3, rangs) == 0.5
    assert evaluation.mrr(rangs) == pytest.approx((1 + 0.5 + 0 + 0.25) / 4)
    assert evaluation.rappel_a(3, []) == 0.0 and evaluation.mrr([]) == 0.0


def test_ce_que_change_le_seuil():
    distances = [[0.10, 0.20], [0.25, 0.30], [0.15, 0.18]]
    rangs = [1, 2, 2]  # question 3 : le bon passage est le 2e, à 0.18
    hors = [[0.30], [0.12], []]
    r = evaluation.au_seuil(0.2, distances, rangs, hors, k=4)
    assert r["repond"] == pytest.approx(2 / 3)
    assert r["bon_passage_garde"] == pytest.approx(2 / 3)
    assert r["refus_hors_corpus"] == pytest.approx(2 / 3)
    assert evaluation.au_seuil(0.2, distances, rangs, hors, k=1)["bon_passage_garde"] == pytest.approx(1 / 3)

"""models/metrics.json : la mesure versionnée de la recherche sur le corpus PIAF complet."""
import json
from pathlib import Path

import pytest

from rag_agent import config

METRIQUES = Path(__file__).resolve().parents[1] / "models" / "metrics.json"
CLES = {"source", "date", "modele_embedding", "n_passages", "n_questions", "rappel_1", "rappel_3",
        "mrr_10", "top_k", "seuil_distance", "repond", "bon_passage_garde", "n_hors_corpus",
        "refus_hors_corpus", "seuil_rappel_3"}


@pytest.fixture(scope="module")
def metriques():
    assert METRIQUES.exists(), "models/metrics.json absent : python src/evaluate.py"
    return json.loads(METRIQUES.read_text(encoding="utf-8"))


def test_toutes_les_cles_sont_la(metriques):
    assert CLES <= set(metriques)


def test_la_mesure_porte_sur_tout_piaf(metriques):
    assert metriques["source"] == "piaf"
    assert metriques["n_passages"] == 761 and metriques["n_questions"] == 3835


def test_la_recherche_passe_le_seuil_d_acceptation(metriques):
    assert metriques["rappel_3"] >= metriques["seuil_rappel_3"]
    assert 0 <= metriques["rappel_1"] <= metriques["rappel_3"] <= 1


def test_la_mesure_correspond_aux_reglages(metriques):
    """Changer le modèle ou le seuil sans réévaluer laisse une mesure fausse : ce test le signale."""
    assert metriques["modele_embedding"] == config.MODELE_EMBEDDING
    assert metriques["seuil_distance"] == config.SEUIL_DISTANCE
    assert metriques["top_k"] == config.TOP_K

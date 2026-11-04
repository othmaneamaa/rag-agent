"""La fiche modèle : chaque rubrique remplie, et les chiffres cités sont ceux du dépôt."""
import json
import re
from pathlib import Path

import pytest

from rag_agent import __version__

RACINE = Path(__file__).resolve().parents[1]
FICHE = RACINE / "docs" / "fiche-modele.md"
METRIQUES = RACINE / "models" / "metrics.json"
RUBRIQUES = ["Identité du modèle", "Usage prévu", "Usages exclus", "Données d'entraînement",
             "Performances mesurées", "Limites connues", "Biais mesurés", "Explicabilité",
             "Conditions d'usage", "Maintenance"]


def rubrique(titre: str) -> str:
    texte = FICHE.read_text(encoding="utf-8")
    m = re.search(rf"^## {re.escape(titre)}.*?$\n(.*?)(?=^## |\Z)", texte, re.M | re.S)
    corps = m.group(1) if m else ""
    return "\n".join(ligne for ligne in corps.splitlines() if ligne.strip() and not ligne.startswith(">"))


def ecrit(valeur: float, texte: str) -> bool:
    nombre = f"{valeur:.3f}"
    return nombre in texte or nombre.replace(".", ",") in texte


@pytest.mark.parametrize("titre", RUBRIQUES)
def test_chaque_rubrique_est_remplie(titre):
    assert len(rubrique(titre)) >= 40, f"rubrique « {titre} » vide ou trop courte"


def test_la_version_citee_est_celle_du_paquet():
    assert __version__ in rubrique("Identité du modèle")


def test_les_performances_citees_sont_celles_du_depot():
    m = json.loads(METRIQUES.read_text(encoding="utf-8"))
    texte = rubrique("Performances mesurées")
    for cle in ["rappel_1", "rappel_3", "mrr_10"]:
        assert ecrit(m[cle], texte), f"citer {cle} = {m[cle]:.3f}"
    assert str(m["seuil_distance"]).replace(".", ",") in texte or str(m["seuil_distance"]) in texte


def test_les_conditions_d_usage_prevoient_un_humain():
    assert "humain" in rubrique("Conditions d'usage").lower()

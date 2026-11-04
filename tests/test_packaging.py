"""Le paquet et la commande rag : version unique, modules propres, ce qui entre dans la roue."""
import re
import tomllib
from pathlib import Path

import pytest

from rag_agent import __version__
from rag_agent.cli import main

RACINE = Path(__file__).resolve().parents[1]


def test_la_version_est_ecrite_a_un_seul_endroit():
    projet = tomllib.loads((RACINE / "pyproject.toml").read_text(encoding="utf-8"))
    assert "version" in projet["project"]["dynamic"]
    assert projet["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "rag_agent.__version__"}
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)


def test_la_roue_ne_contient_que_le_code():
    projet = tomllib.loads((RACINE / "pyproject.toml").read_text(encoding="utf-8"))
    assert projet["tool"]["setuptools"]["packages"] == ["rag_agent", "api", "ui"]
    assert projet["project"]["scripts"] == {"rag": "rag_agent.cli:main"}


@pytest.mark.parametrize("dossier", ["api", "rag_agent"])
def test_le_code_installe_ne_bricole_pas_le_chemin_de_python(dossier):
    for fichier in (RACINE / dossier).glob("*.py"):
        assert "sys.path" not in fichier.read_text(encoding="utf-8"), fichier


def test_rag_version(capsys):
    with pytest.raises(SystemExit) as fin:
        main(["--version"])
    assert fin.value.code == 0
    assert capsys.readouterr().out.strip() == f"rag-agent {__version__}"


def test_rag_search(monkeypatch, capsys, index):
    from rag_agent import cli, config

    monkeypatch.setattr(cli, "_index", lambda: index)
    monkeypatch.setattr(config, "SEUIL_DISTANCE", 0.95)
    assert main(["search", "capitale", "de", "la", "Russie", "-k", "1"]) == 0
    assert "[Moscou, P" in capsys.readouterr().out


def test_rag_search_sans_passage_annonce_le_refus(monkeypatch, capsys, index):
    from rag_agent import cli, config

    monkeypatch.setattr(cli, "_index", lambda: index)
    monkeypatch.setattr(config, "SEUIL_DISTANCE", 0.1)
    assert main(["search", "pâte", "à", "crêpes"]) == 0
    assert "l'agent refuserait" in capsys.readouterr().out


def test_rag_sans_index_explique_et_sort_en_erreur(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("RAG_INDEX", str(tmp_path / "rien"))
    assert main(["search", "capitale"]) == 1
    assert "index introuvable" in capsys.readouterr().err


def test_rag_ask_sans_cle_explique_et_sort_en_erreur(monkeypatch, capsys, index):
    from rag_agent import cli, config, llm

    monkeypatch.setattr(cli, "_index", lambda: index)
    monkeypatch.setattr(config, "SEUIL_DISTANCE", 0.95)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    llm._client.cache_clear()
    assert main(["ask", "capitale"]) == 1
    assert "GROQ_API_KEY absente" in capsys.readouterr().err

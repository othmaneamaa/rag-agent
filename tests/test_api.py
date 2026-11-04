"""L'API : contrat d'entrée, réponses, refus, et codes d'erreur sans fuite d'information."""
import pytest
from fastapi.testclient import TestClient

import api.main
from api.agent import Agent
from rag_agent.llm import LLMNonConfigure


def test_health(client_api):
    r = client_api.get("/health")
    assert r.status_code == 200
    corps = r.json()
    assert corps["status"] == "OK" and corps["passages"] == 4
    assert corps["modele_embedding"] == "hachage-test"


def test_accueil(client_api):
    assert client_api.get("/").json()["docs"] == "/docs"


def test_search_renvoie_les_passages_sans_llm(client_api, faux_llm):
    r = client_api.post("/search", json={"question": "capitale de la Russie", "k": 2})
    assert r.status_code == 200
    passages = r.json()["passages"]
    assert len(passages) == 2 and passages[0]["titre"] == "Moscou"
    assert faux_llm.appels == []


def test_ask_repond_avec_ses_sources(client_api):
    r = client_api.post("/ask", json={"question": "Où est né le chimiste Lavoisier ?"})
    assert r.status_code == 200
    corps = r.json()
    assert corps["statut"] == "REPONSE"
    assert corps["sources"][0]["titre"] == "Antoine Lavoisier"


def test_ask_refuse_hors_corpus(client_api, monkeypatch, faux_llm):
    from rag_agent import config

    monkeypatch.setattr(config, "SEUIL_DISTANCE", 0.3)
    r = client_api.post("/ask", json={"question": "recette de la pâte à crêpes"})
    assert r.status_code == 200
    assert r.json()["statut"] == "REFUS" and r.json()["sources"] == []
    assert faux_llm.appels == []


@pytest.mark.parametrize(
    "corps",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "\n\t"},
        {"question": "x" * 501},
        {"question": 42},
        {"question": "capitale ?", "k": 0},
        {"question": "capitale ?", "k": 11},
    ],
    ids=["absente", "vide", "espaces", "blancs", "trop-longue", "pas-un-texte", "k-zero", "k-trop-grand"],
)
def test_les_questions_hors_contrat_recoivent_422(client_api, corps):
    for route in ["/ask", "/search"]:
        assert client_api.post(route, json=corps).status_code == 422


def test_la_question_de_500_caracteres_passe(client_api):
    assert client_api.post("/search", json={"question": "capitale " + "x" * 491}).status_code == 200


def _api_avec(monkeypatch, index, generer):
    monkeypatch.setattr(api.main, "charger_agent", lambda: Agent(index, generer))
    from rag_agent import config

    monkeypatch.setattr(config, "SEUIL_DISTANCE", 0.95)
    return TestClient(api.main.app)


def test_sans_cle_groq_ask_renvoie_503_et_search_marche(monkeypatch, index):
    def sans_cle(prompt):
        raise LLMNonConfigure("GROQ_API_KEY absente")

    with _api_avec(monkeypatch, index, sans_cle) as client:
        r = client.post("/ask", json={"question": "capitale de la Russie"})
        assert r.status_code == 503 and "Groq" in r.json()["detail"]
        assert client.post("/search", json={"question": "capitale de la Russie"}).status_code == 200


@pytest.mark.parametrize("exception", [RuntimeError("quota Groq dépassé, clé gsk_secret"), ValueError("bug")])
def test_une_panne_du_llm_renvoie_502_sans_message_interne(monkeypatch, index, exception):
    def en_panne(prompt):
        raise exception

    with _api_avec(monkeypatch, index, en_panne) as client:
        r = client.post("/ask", json={"question": "capitale de la Russie"})
    assert r.status_code == 502
    assert "gsk_" not in r.text and "quota" not in r.text and "bug" not in r.text


def test_une_panne_de_la_recherche_renvoie_500(client_api, monkeypatch):
    def en_panne(*args, **kwargs):
        raise RuntimeError("chromadb corrompu")

    monkeypatch.setattr(api.main.app.state.agent, "chercher", en_panne)
    r = client_api.post("/search", json={"question": "capitale"})
    assert r.status_code == 500 and "chromadb" not in r.text


def test_sans_index_l_api_repond_503(client_api, monkeypatch):
    monkeypatch.setattr(api.main.app.state, "agent", None)
    assert client_api.get("/health").status_code == 503
    assert client_api.post("/ask", json={"question": "capitale"}).status_code == 503


def test_l_api_refuse_de_demarrer_sans_index(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_INDEX", str(tmp_path / "rien"))
    with pytest.raises(FileNotFoundError, match="index introuvable"):
        with TestClient(api.main.app):
            pass


def test_le_nom_du_llm_dit_s_il_est_configure(monkeypatch, index):
    from rag_agent import llm

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert Agent(index).nom_llm == "non configuré"
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    assert Agent(index).nom_llm == "openai/gpt-oss-120b"
    assert llm.configure()

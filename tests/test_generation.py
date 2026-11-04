"""La réponse sourcée et les deux refus, avec un faux modèle de langage."""
from rag_agent.generation import MARQUEUR_REFUS, MESSAGE_REFUS, construire_prompt, repondre


def test_sans_passage_l_agent_refuse_sans_appeler_le_llm(index, faux_llm):
    resultat = repondre("recette de la pâte à crêpes", index, faux_llm, seuil=0.3)
    assert resultat["statut"] == "REFUS"
    assert resultat["reponse"] == MESSAGE_REFUS
    assert resultat["sources"] == []
    assert faux_llm.appels == []


def test_une_reponse_cite_ses_sources(index, faux_llm):
    resultat = repondre("Où est né le chimiste Lavoisier ?", index, faux_llm, k=2, seuil=0.95)
    assert resultat["statut"] == "REPONSE"
    assert len(faux_llm.appels) == 1
    assert [s["titre"] for s in resultat["sources"]] == ["Antoine Lavoisier"]
    assert resultat["sources"][0]["passage_id"] in resultat["reponse"]
    assert len(resultat["passages"]) == 2


def test_le_marqueur_hors_corpus_devient_un_refus(index):
    for sortie in [MARQUEUR_REFUS, f" {MARQUEUR_REFUS}. ", MARQUEUR_REFUS.lower(), ""]:
        resultat = repondre("capitale", index, lambda prompt, s=sortie: s, seuil=0.95)
        assert resultat["statut"] == "REFUS", sortie
        assert resultat["passages"], "les passages consultés restent visibles"


def test_sans_identifiant_cite_tous_les_passages_servent_de_sources(index):
    resultat = repondre("capitale", index, lambda prompt: "Une réponse sans citation.", k=3, seuil=0.95)
    assert resultat["statut"] == "REPONSE"
    assert len(resultat["sources"]) == len(resultat["passages"])


def test_le_prompt_borne_le_modele_aux_extraits(index):
    passages = index.chercher("capitale de la Russie", k=2, seuil=None)
    prompt = construire_prompt("Quelle est la capitale de la Russie ?", passages)
    assert "UNIQUEMENT" in prompt and MARQUEUR_REFUS in prompt
    for p in passages:
        assert f"[{p['titre']}, {p['passage_id']}]" in prompt
        assert p["texte"] in prompt
    assert prompt.rstrip().endswith("Réponse :")

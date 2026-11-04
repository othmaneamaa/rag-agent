"""Le client HTTP de l'interface : chaque réponse de l'API devient un message lisible."""
import httpx
import pytest

from ui import client


def session(statut, corps=None):
    def repondre(requete):
        return httpx.Response(statut, json=corps if corps is not None else {})

    return httpx.Client(base_url="http://test", transport=httpx.MockTransport(repondre))


def test_une_reponse_passe_telle_quelle():
    corps = {"statut": "REPONSE", "reponse": "Né en 1743 [Antoine Lavoisier, P1]", "sources": [], "passages": []}
    r = client.demander(session(200, corps), "Quand Lavoisier est-il né ?")
    assert r.ok and not r.refus and r.corps == corps


def test_un_refus_est_reconnu():
    r = client.demander(session(200, {"statut": "REFUS", "reponse": "...", "sources": [], "passages": []}), "?")
    assert r.ok and r.refus


@pytest.mark.parametrize(
    ("statut", "corps", "message"),
    [
        (422, {"detail": []}, client.REFUSEE),
        (503, {"detail": "Génération indisponible : clé Groq absente"}, client.SANS_LLM),
        (503, {"detail": "Index non chargé"}, client.SANS_INDEX),
        (502, {"detail": "Service de génération indisponible"}, client.GENERATION),
    ],
)
def test_les_erreurs_sont_traduites(statut, corps, message):
    r = client.demander(session(statut, corps), "question")
    assert not r.ok and r.message == message


def test_une_erreur_inattendue_donne_son_code():
    assert client.chercher(session(500), "q").message == "Erreur interne de l'API (500)."


def test_une_api_eteinte_est_signalee():
    def coupe(requete):
        raise httpx.ConnectError("refusé")

    eteinte = httpx.Client(base_url="http://test", transport=httpx.MockTransport(coupe))
    assert client.sante(eteinte).message == client.INJOIGNABLE


def test_le_client_parle_a_la_vraie_api(client_api):
    """Bout à bout : le client, branché sur l'API de test, sans réseau."""
    assert client.sante(client_api).corps["passages"] == 4
    r = client.demander(client_api, "Où est né le chimiste Lavoisier ?")
    assert r.ok and r.corps["sources"][0]["titre"] == "Antoine Lavoisier"
    assert client.demander(client_api, "   ").message == client.REFUSEE


def test_une_citation_lisible():
    assert client.citation({"titre": "Moscou", "passage_id": "P12345678", "distance": 0.1234}) == (
        "Moscou · P12345678 · distance 0.123"
    )

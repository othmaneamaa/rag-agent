"""Page Streamlit de l'agent de recherche documentaire.

La page est un client de l'API : elle n'importe jamais l'index ni le modèle de langage.
Tout passe par ui/client.py.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # pour importer ui.client

import streamlit as st

from ui import client

st.set_page_config(page_title="Agent de recherche documentaire", page_icon="🔎", layout="wide")
st.title("Agent de recherche documentaire")
st.caption("Corpus : 761 paragraphes de Wikipédia en français, jeu PIAF publié sur Kaggle.")

session = client.creer_session()
etat = client.sante(session)
if not etat.ok:
    st.error(etat.message)
    st.stop()
with st.sidebar:
    st.subheader("L'agent")
    st.metric("Passages indexés", etat.corps["passages"])
    st.metric("Seuil de distance", etat.corps["seuil"])
    st.caption(f"Embedding : `{etat.corps['modele_embedding']}`")
    st.caption(f"Modèle de langage : `{etat.corps['llm']}`")
    k = st.slider("Passages récupérés", 1, 10, 4)

onglet_question, onglet_recherche, onglet_mesures = st.tabs(
    ["Poser une question", "Chercher des passages", "Mesures"]
)

with onglet_question:
    question = st.text_input("Votre question", placeholder="Par exemple : Quand Lavoisier est-il né ?")
    if st.button("Demander", type="primary"):
        with st.spinner("Recherche et rédaction..."):
            reponse = client.demander(session, question, k)
        if not reponse.ok:
            st.warning(reponse.message)
        elif reponse.refus:
            st.info(f"{reponse.corps['reponse']} L'agent préfère ne rien inventer.")
        else:
            st.markdown(reponse.corps["reponse"])
            st.subheader("Sources")
            for source in reponse.corps["sources"]:
                st.markdown(f"- {client.citation(source)}")
        if reponse.ok and reponse.corps.get("passages"):
            with st.expander("Passages transmis au modèle de langage"):
                for p in reponse.corps["passages"]:
                    st.markdown(f"**{client.citation(p)}**")
                    st.write(p["texte"])

with onglet_recherche:
    st.caption("La recherche seule, sans rédaction : utile sans clé Groq, et pour comprendre le seuil.")
    recherche = st.text_input("Question", key="recherche")
    if st.button("Chercher"):
        reponse = client.chercher(session, recherche, k)
        if not reponse.ok:
            st.warning(reponse.message)
        elif not reponse.corps["passages"]:
            st.info(f"Aucun passage sous le seuil de {reponse.corps['seuil']} : l'agent refuserait.")
        else:
            for p in reponse.corps["passages"]:
                st.markdown(f"**{client.citation(p)}**")
                st.write(p["texte"])

with onglet_mesures:
    chemin = Path(__file__).resolve().parents[1] / "models" / "metrics.json"
    if chemin.exists():
        m = json.loads(chemin.read_text(encoding="utf-8"))
        st.caption(f"Évaluation du {m['date']} sur {m['n_questions']} questions ({m['modele_embedding']}).")
        a, b, c = st.columns(3)
        a.metric("Bon passage en 1re position", f"{m['rappel_1']:.1%}")
        b.metric("Bon passage dans le top 3", f"{m['rappel_3']:.1%}")
        c.metric("MRR@10", f"{m['mrr_10']:.3f}")
        d, e = st.columns(2)
        d.metric(f"Questions du corpus traitées (seuil {m['seuil_distance']})", f"{m['repond']:.1%}")
        e.metric("Questions hors corpus refusées sans LLM", f"{m['refus_hors_corpus']:.1%}")
    else:
        st.info("models/metrics.json absent : python src/evaluate.py le produit.")

"""Interface Streamlit de l'agent de recherche documentaire."""
import streamlit as st

from helpers import charger_agent, poser_question, formater_citation

st.set_page_config(
    page_title="Agent de recherche documentaire",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Agent de recherche documentaire")
st.caption(
    "Posez une question en langage naturel : l'agent cherche dans le corpus "
    "indexé, rédige une réponse et cite ses sources."
)

# Chargement de l'agent (une seule fois grâce au cache Streamlit)
@st.cache_resource
def agent():
    return charger_agent()

try:
    repondre = agent()
except Exception as e:
    st.error(f"Impossible de charger l'agent : {e}")
    st.info("Vérifiez que `.env` contient `GROQ_API_KEY` et que le corpus est indexé.")
    st.stop()

# Trois onglets : question, corpus, explication
onglet_question, onglet_corpus, onglet_aide = st.tabs(
    ["Poser une question", "Explorer le corpus", "Comment ça marche"]
)

# ---------- Onglet 1 : question ----------
with onglet_question:
    question = st.text_input(
        "Votre question :",
        placeholder="Par exemple : Qu'est-ce que le RAG ?",
    )
    bouton = st.button("Rechercher", type="primary")

    if bouton and question:
        with st.spinner("Recherche et rédaction en cours..."):
            try:
                resultat = poser_question(question, repondre)
            except ValueError as e:
                st.warning(str(e))
                st.stop()

        st.subheader("Réponse")
        st.write(resultat["reponse"])

        st.subheader("Sources")
        for i, source in enumerate(resultat["sources_uniques"], 1):
            st.markdown(f"**[{i}]** `{formater_citation(source)}`")

        with st.expander("Voir les extraits utilisés"):
            for i, p in enumerate(resultat["passages"], 1):
                st.markdown(
                    f"**[Extrait {i}]** `{formater_citation(p)}` "
                    f"(distance {p['distance']:.3f})"
                )
                st.code(p["texte"][:500] + ("..." if len(p["texte"]) > 500 else ""))

# ---------- Onglet 2 : corpus ----------
with onglet_corpus:
    st.subheader("Documents indexés")
    st.info(
        "Pour ajouter un document : placez-le dans `data/pdfs/`, "
        "puis relancez `python src/ingest.py`."
    )
    try:
        from pathlib import Path
        dossier = Path(__file__).resolve().parents[1] / "data" / "pdfs"
        pdfs = sorted(dossier.glob("*.pdf"))
        if pdfs:
            for pdf in pdfs:
                st.markdown(f"- 📄 `{pdf.name}`")
        else:
            st.warning("Aucun PDF dans `data/pdfs/`.")
    except Exception as e:
        st.warning(f"Impossible de lister les documents : {e}")

# ---------- Onglet 3 : aide ----------
with onglet_aide:
    st.subheader("Comment fonctionne cet agent ?")
    st.markdown(
        """
        1. **Ingestion** — les PDF sont découpés en passages (512 tokens,
           chevauchement 50) et convertis en vecteurs via
           `sentence-transformers`.
        2. **Recherche** — à chaque question, l'agent encode la question et
           récupère les passages les plus proches dans ChromaDB.
        3. **Génération** — les passages sont fournis à un LLM (Groq,
           `openai/gpt-oss-120b`) qui rédige une réponse **ancrée dans les
           extraits**, en citant les sources.

        **Pourquoi un RAG ?** Pour limiter les hallucinations : le modèle ne
        répond qu'à partir des documents fournis. Si l'information n'y est pas,
        il le dit.
        """
    )
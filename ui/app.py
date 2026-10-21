"""Interface Streamlit de l'agent de recherche documentaire."""
import csv
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

from helpers import charger_agent, poser_question, formater_citation

RACINE = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="Agent de recherche documentaire",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Sidebar ----------
with st.sidebar:
    st.title("🔍 Agent RAG")
    st.caption("Recherche documentaire augmentée")
    st.divider()

    st.subheader("Corpus")
    dossier = RACINE / "data" / "pdfs"
    pdfs = sorted(dossier.glob("*.pdf")) if dossier.exists() else []
    st.metric("Documents", len(pdfs))
    st.metric("Passages indexés", "508")
    st.metric("Seuil de similarité", "0.55")

    st.divider()
    st.caption("Modèle : `openai/gpt-oss-120b` (Groq)")
    st.caption("Embedding : `all-MiniLM-L6-v2`")

# ---------- Chargement de l'agent ----------
@st.cache_resource
def agent():
    return charger_agent()

try:
    repondre = agent()
except Exception as e:
    st.error(f"Impossible de charger l'agent : {e}")
    st.info("Vérifiez que `.env` contient `GROQ_API_KEY` et que le corpus est indexé.")
    st.stop()

# ---------- Titre ----------
st.title("🔍 Agent de recherche documentaire")
st.caption(
    "Posez une question en langage naturel : l'agent cherche dans le corpus "
    "indexé, rédige une réponse et cite ses sources."
)

# ---------- Onglets ----------
onglet_question, onglet_corpus, onglet_evaluation, onglet_aide = st.tabs(
    ["💬 Poser une question", "📚 Explorer le corpus", "📊 Évaluation", "ℹ️ Comment ça marche"]
)

# ---------- Onglet 1 : question ----------
with onglet_question:
    question = st.text_input(
        "Votre question :",
        placeholder="Par exemple : Qu'est-ce que la régularisation ?",
    )
    bouton = st.button("Rechercher", type="primary")

    if bouton:
        if not question or not question.strip():
            st.warning("Saisissez une question avant de cliquer sur Rechercher.")
        else:
            with st.spinner("Recherche et rédaction en cours..."):
                debut = time.time()
                try:
                    resultat = poser_question(question, repondre)
                except ValueError as e:
                    st.warning(str(e))
                    st.stop()
                duree = time.time() - debut

            st.subheader("Réponse")
            if resultat["reponse"] == "Aucun document pertinent trouvé.":
                st.info("Aucun passage pertinent n'a été trouvé sous le seuil de similarité.")
            else:
                st.markdown(resultat["reponse"])

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sources", len(resultat["sources_uniques"]))
            with col2:
                st.metric("Passages", len(resultat["passages"]))
            with col3:
                st.metric("Temps", f"{duree:.2f} s")

            if resultat["sources_uniques"]:
                st.subheader("Sources")
                for i, source in enumerate(resultat["sources_uniques"], 1):
                    st.markdown(f"**[{i}]** `{formater_citation(source)}`")

            if resultat["passages"]:
                with st.expander("Voir les passages récupérés", expanded=False):
                    for i, p in enumerate(resultat["passages"], 1):
                        st.markdown(
                            f"**[Extrait {i}]** `{formater_citation(p)}` "
                            f"— distance **{p['distance']:.3f}**"
                        )
                        st.code(
                            p["texte"][:500] + ("..." if len(p["texte"]) > 500 else ""),
                            language="text",
                        )

# ---------- Onglet 2 : corpus ----------
with onglet_corpus:
    st.subheader("Documents indexés")
    st.info(
        "Pour ajouter un document : placez-le dans `data/pdfs/`, "
        "puis relancez `python src/ingest.py`."
    )
    if pdfs:
        for pdf in pdfs:
            taille_ko = pdf.stat().st_size / 1024
            st.markdown(f"- 📄 `{pdf.name}` ({taille_ko:.0f} Ko)")
    else:
        st.warning("Aucun PDF dans `data/pdfs/`.")

# ---------- Onglet 3 : évaluation ----------
with onglet_evaluation:
    st.subheader("Évaluation quantitative")
    st.caption(
        "20 questions de référence, dont 15 dans le corpus et 5 hors corpus. "
        "Le script `src/evaluate_rag.py` mesure la Context Precision et le taux de refus."
    )

    if st.button("Lancer l'évaluation", type="primary"):
        with st.spinner("Évaluation en cours (peut prendre 1 à 2 minutes)..."):
            resultat = subprocess.run(
                [sys.executable, "src/evaluate_rag.py"],
                capture_output=True,
                text=True,
                cwd=RACINE,
            )
        if resultat.returncode != 0:
            st.error("L'évaluation a échoué.")
            st.code(resultat.stderr, language="text")
        else:
            st.success("Évaluation terminée.")
            st.code(resultat.stdout, language="text")

    st.divider()

    st.subheader("Jeu de test")
    chemin = RACINE / "data" / "questions_reference.csv"
    if chemin.exists():
        with chemin.open(encoding="utf-8", newline="") as f:
            questions = list(csv.DictReader(f))
        st.dataframe(
            [
                {
                    "Question": q["question"],
                    "Source attendue": q["source_attendue"] or "—",
                    "Page attendue": q["page_attendue"] or "—",
                }
                for q in questions
            ],
            use_container_width=True,
        )
    else:
        st.warning("Fichier `data/questions_reference.csv` introuvable.")

# ---------- Onglet 4 : aide ----------
with onglet_aide:
    st.subheader("Comment fonctionne cet agent ?")
    st.markdown(
        """
        Cet agent suit le pattern **RAG** (Retrieval-Augmented Generation), en 3 étapes :

        **1. Ingestion**
        Les PDF sont découpés en passages (512 tokens, chevauchement 50) et convertis
        en vecteurs via `sentence-transformers/all-MiniLM-L6-v2`.

        **2. Recherche**
        À chaque question, l'agent encode la question et récupère les passages les plus
        proches dans ChromaDB. Un seuil de similarité (0.55) filtre les passages hors sujet.

        **3. Génération**
        Les passages sont fournis à un LLM (Groq, `openai/gpt-oss-120b`) qui rédige une
        réponse **ancrée dans les extraits**, en citant les sources.

        **Pourquoi un RAG ?**
        Pour limiter les hallucinations : le modèle ne répond qu'à partir des documents
        fournis. Si l'information n'y est pas, il le dit explicitement.
        """
    )
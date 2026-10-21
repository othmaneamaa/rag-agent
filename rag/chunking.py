"""Chargement des PDF et découpage en passages."""
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import PDFS, CHUNK_SIZE, CHUNK_OVERLAP

# Marqueurs de pages non informatives (pages de garde, remerciements, liens, etc.)
MOTS_INUTILES = [
    "sommaire",
    "table des matières",
    "table des matieres",
    "remerciements",
    "suivez-nous",
    "suivez nous",
    "linkedin",
    "bibliographie",
    "page de garde",
    "copyright",
    "tous droits réservés",
    "tous droits reserves",
]

# En dessous de ce nombre de caractères, la page est considérée comme vide ou non informative
LONGUEUR_MINIMALE = 100


def page_utile(texte: str) -> bool:
    """Renvoie False si la page est manifestement non informative."""
    texte_bas = texte.lower().strip()
    if len(texte_bas) < LONGUEUR_MINIMALE:
        return False
    for mot in MOTS_INUTILES:
        if mot in texte_bas:
            return False
    return True


def charger_documents(dossier: Path = PDFS):
    """Charge tous les PDF d'un dossier en documents LangChain, en ignorant les pages non informatives."""
    documents = []
    pages_ignorees = 0
    for pdf in sorted(dossier.glob("*.pdf")):
        loader = PyPDFLoader(str(pdf))
        for page in loader.load():
            texte = re.sub(r"\s+", " ", page.page_content).strip()
            if not page_utile(texte):
                pages_ignorees += 1
                continue
            page.page_content = texte
            page.metadata["source"] = pdf.name
            page.metadata["page"] = page.metadata.get("page", 0) + 1
            documents.append(page)
    if pages_ignorees:
        print(f"   {pages_ignorees} page(s) ignorée(s) (garde, remerciements, liens…)")
    return documents


def decouper(documents):
    """Découpe les documents en passages de taille fixe avec chevauchement."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(documents)
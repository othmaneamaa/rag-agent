"""Chargement des PDF et découpage en passages."""
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import PDFS, CHUNK_SIZE, CHUNK_OVERLAP


def charger_documents(dossier: Path = PDFS):
    """Charge tous les PDF d'un dossier en documents LangChain."""
    documents = []
    for pdf in sorted(dossier.glob("*.pdf")):
        loader = PyPDFLoader(str(pdf))
        for page in loader.load():
            texte = re.sub(r"\s+", " ", page.page_content).strip()
            page.page_content = texte
            page.metadata["source"] = pdf.name
            page.metadata["page"] = page.metadata.get("page", 0) + 1
            documents.append(page)
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
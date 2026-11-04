"""Les mesures de la recherche, en fonctions pures : testables sans index ni modèle.

- rappel@k : la question retrouve-t-elle, parmi ses k premiers passages, celui dont elle est tirée ?
- MRR : l'inverse du rang du bon passage, en moyenne (1 s'il est premier, 1/2 s'il est deuxième...).
- au seuil s : part des questions du corpus qui gardent au moins un passage (l'agent répond),
  part des questions hors corpus qui n'en gardent aucun (l'agent refuse sans appeler le LLM).
"""
from __future__ import annotations

from collections.abc import Sequence


def rang(attendu: str, trouves: Sequence[str]) -> int:
    """Rang (1, 2, ...) du passage attendu parmi les trouvés ; 0 s'il n'y est pas."""
    return trouves.index(attendu) + 1 if attendu in trouves else 0


def rappel_a(k: int, rangs: Sequence[int]) -> float:
    if not rangs:
        return 0.0
    return sum(1 for r in rangs if 0 < r <= k) / len(rangs)


def mrr(rangs: Sequence[int]) -> float:
    if not rangs:
        return 0.0
    return sum(1 / r for r in rangs if r > 0) / len(rangs)


def au_seuil(
    seuil: float,
    distances_corpus: Sequence[Sequence[float]],
    rangs_corpus: Sequence[int],
    distances_hors: Sequence[Sequence[float]],
    k: int,
) -> dict[str, float]:
    """Ce que change un seuil : réponses données, bon passage gardé, refus hors corpus.

    distances_* : pour chaque question, les distances de ses passages, triées (la plus proche d'abord).
    rangs_corpus : le rang du bon passage parmi ces passages (0 s'il n'y est pas).
    """
    repond = [bool(d) and d[0] <= seuil for d in distances_corpus]
    bon_garde = [
        0 < r <= k and d[r - 1] <= seuil for r, d in zip(rangs_corpus, distances_corpus, strict=True)
    ]
    refus = [not d or d[0] > seuil for d in distances_hors]
    n = max(len(distances_corpus), 1)
    return {
        "seuil": seuil,
        "repond": sum(repond) / n,
        "bon_passage_garde": sum(bon_garde) / n,
        "refus_hors_corpus": sum(refus) / max(len(distances_hors), 1),
    }

# codon_distances.py

from typing import Iterable

from thefuzz import fuzz

__all__ = [
    'generate_distances'
]

ACIDS = ['A', 'T', 'G', 'C']

def generate_distances(codons: Iterable[str]) -> dict[str, dict[str, float]]:

    return {
        codon1: dict(
            sorted(
                {
                    codon2: fuzz.ratio(" ".join(codon1), " ".join(codon2)) / 100
                    for codon2 in codons
                    if codon1 != codon2
                }.items(),
                key=lambda data: data[1],
                reverse=True
            )
        )
        for codon1 in codons
    }

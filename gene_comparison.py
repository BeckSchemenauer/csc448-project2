from __future__ import annotations
from itertools import combinations
from typing import Dict, Set, Tuple
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

def _read_gene_set(path: str) -> Set[str]:
    """Read one-gene-per-line .txt into a cleaned set."""
    genes: Set[str] = set()
    with open(path, "r") as f:
        for line in f:
            g = line.strip()
            if not g:
                continue
            genes.add(g)
    return genes


def jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity = |A∩B| / |A∪B|."""
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def venn_and_jaccard_3(
    file_a: str,
    file_b: str,
    file_c: str,
    labels: Tuple[str, str, str] = ("Method 1", "Method 2", "Authors"),
    output_pdf: str = "venn_methods.pdf",
):
    """
    Make a 3-way Venn diagram from three gene-list .txt files and print
    pairwise Jaccard similarities.

    Parameters
    ----------
    file_a, file_b, file_c : str
        Paths to one-gene-per-line txt files.
    labels : (str, str, str)
        Labels for the Venn sets in the same order as files.
    output_pdf : str
        Where to save the Venn diagram PDF.
    """
    sets: Dict[str, Set[str]] = {
        labels[0]: _read_gene_set(file_a),
        labels[1]: _read_gene_set(file_b),
        labels[2]: _read_gene_set(file_c),
    }

    # --- Print sizes ---
    for name, s in sets.items():
        print(f"{name}: {len(s)} genes")

    # --- Pairwise Jaccard + overlap counts ---
    print("\nPairwise overlap + Jaccard:")
    for (name1, s1), (name2, s2) in combinations(sets.items(), 2):
        inter = len(s1 & s2)
        union = len(s1 | s2)
        jac = jaccard(s1, s2)
        print(f"- {name1} vs {name2}: |∩|={inter}, |∪|={union}, Jaccard={jac:.4f}")

    # --- Venn diagram ---
    plt.figure(figsize=(7, 6))
    venn3([sets[labels[0]], sets[labels[1]], sets[labels[2]]], set_labels=labels)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_pdf, format="pdf", dpi=600, bbox_inches="tight")
    plt.close()

    print(f"\nSaved Venn diagram to '{output_pdf}'")

    return sets

venn_and_jaccard_3(
    file_a="top_230_method1.txt",
    file_b="top_230_method2.txt",
    file_c="top_230_author.txt",
    labels=("Method 1", "Method 2", "Authors"),
    output_pdf="venn_3way.pdf",
)

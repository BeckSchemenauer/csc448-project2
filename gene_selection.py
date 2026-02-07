import pandas as pd
import numpy as np

def extract_author_230(
    input_file: str = "data/230_authors.txt",
    output_file: str = "top_230_author.txt"
):
    """
    Extract gene names from the authors' 230-gene list and
    write one gene name per line to a text file.

    Parameters
    ----------
    input_file : str
        Path to the authors' gene list file.
    output_file : str
        Output file path.
    """

    # Read tab-delimited file
    df = pd.read_csv(input_file, sep="\t")

    # Drop missing or empty gene names
    genes = (
        df["ORF"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    # Write one gene per line
    with open(output_file, "w") as f:
        for gene in genes:
            f.write(f"{gene}\n")

    print(f"Saved {len(genes)} genes to '{output_file}'")

    return genes.tolist()


def select_top_genes_min_max(
    df: pd.DataFrame,
    log_cols: list[str],
    gene_col: str = "ORF",
    n_genes: int = 230,
    output_file: str = "top_230_method1.txt"
):

    # --- Define early and late phase columns ---
    early_cols = log_cols[0:3]   # time points 1–3
    late_cols  = log_cols[4:7]   # time points 5–7

    scores = []

    for _, row in df.iterrows():
        early_vals = row[early_cols].values.astype(float)
        late_vals = row[late_cols].values.astype(float)

        # Ignore genes with missing values in either phase
        if np.any(np.isnan(early_vals)) or np.any(np.isnan(late_vals)):
            continue

        score = max(
            abs(np.min(early_vals) - np.max(late_vals)),
            abs(np.max(early_vals) - np.min(late_vals))
        )

        scores.append((row[gene_col], score))

    # --- Rank genes ---
    scores.sort(key=lambda x: x[1], reverse=True)
    top_genes = scores[:n_genes]

    # --- Write gene names to file ---
    with open(output_file, "w") as f:
        for gene, _ in top_genes:
            f.write(f"{gene}\n")

    print(f"Saved top {n_genes} genes to '{output_file}'")

    return top_genes

def select_genes_by_mean_shift(
    df: pd.DataFrame,
    log_cols: list[str],
    gene_col: str = "ORF",
    top_gain_n: int = 115,
    top_drop_n: int = 115,
    output_file: str = "top_230_method2.txt",
):
    """
    Mean-shift metric:
        early_mean = mean(timepoints 1–3)
        late_mean  = mean(timepoints 5–7)
        delta      = late_mean - early_mean

    Selection:
      - top_abs_n genes by |delta|
      - top_gain_n genes by largest positive delta
      - top_drop_n genes by most negative delta

    Writes one gene name per row to a single output file.
    Returns a list of selected gene names.
    """

    early_cols = log_cols[0:3]   # timepoints 1–3
    late_cols  = log_cols[4:7]   # timepoints 5–7 (skip middle)

    X_early = df[early_cols].to_numpy(dtype=float)
    X_late  = df[late_cols].to_numpy(dtype=float)

    valid = (~np.isnan(X_early).any(axis=1)) & (~np.isnan(X_late).any(axis=1))
    early_mean = X_early[valid].mean(axis=1)
    late_mean  = X_late[valid].mean(axis=1)

    delta = late_mean - early_mean

    res = pd.DataFrame({
        "gene": df.loc[valid, gene_col].values,
        "delta": delta,
        "abs_delta": np.abs(delta)
    })

    # Rank subsets
    top_gain = res.sort_values("delta", ascending=False).head(top_gain_n)
    top_drop = res.sort_values("delta", ascending=True).head(top_drop_n)

    # Combine, preserving priority: abs → gain → drop
    combined = pd.concat([top_gain, top_drop], axis=0)

    # Drop duplicates while preserving order
    combined_genes = combined["gene"].drop_duplicates().tolist()

    # Sanity check
    if len(combined_genes) != top_gain_n + top_drop_n:
        print(f"Warning: duplicates detected; final list contains {len(combined_genes)} genes")

    # Write output
    with open(output_file, "w") as f:
        for gene in combined_genes:
            f.write(f"{gene}\n")

    print(f"Saved {len(combined_genes)} genes to '{output_file}'")

    return combined_genes
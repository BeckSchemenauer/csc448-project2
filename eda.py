import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm


def load_and_transform(file_path: str = "Data/diauxic_raw_ratios.txt"):
    df = pd.read_csv(file_path, sep="\t")

    ratio_cols = [col for col in df.columns if "Ratio" in col]

    # Log2 transform
    log_df = df.copy()
    for col in ratio_cols:
        log_df[f"Log2_{col}"] = np.log2(df[col].replace(0, np.nan))

    log_cols = [f"Log2_{c}" for c in ratio_cols]

    # collapse duplicate ORFs
    if "ORF" in log_df.columns:
        # Compute per-gene variability (std across time)
        log_df["_tmp_std"] = log_df[log_cols].std(axis=1, skipna=True)

        # Keep the most variable instance of each ORF
        log_df = (
            log_df
            .sort_values("_tmp_std", ascending=False)
            .drop_duplicates(subset="ORF", keep="first")
            .drop(columns="_tmp_std")
            .reset_index(drop=True)
        )

    return df, log_df, ratio_cols


def plot_distribution(df_raw, df_log, df_z, ratio_cols, output_name="distribution_plots.pdf"):
    sns.set_context("talk")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))  # Changed to 3 subplots

    # Flatten and clean data
    raw_vals = df_raw[ratio_cols].values.flatten()
    log_vals = df_log[[f"Log2_{c}" for c in ratio_cols]].values.flatten()
    z_vals = df_z[[f"Z_Log2_{c}" for c in ratio_cols]].values.flatten()

    raw_vals = raw_vals[~np.isnan(raw_vals)]
    log_vals = log_vals[~np.isnan(log_vals)]
    z_vals = z_vals[~np.isnan(z_vals)]

    # Raw Ratios Plot
    sns.histplot(raw_vals, kde=False, ax=axes[0], color='#154734',
                 edgecolor='black', linewidth=0.5, binwidth=0.1)
    # axes[0].set_title("Distribution: Raw Ratios") # Commented for Overleaf
    axes[0].set_xlabel("Ratio Value")

    # Log2 Plot
    sns.histplot(log_vals, kde=False, ax=axes[1], color='#154734',
                 edgecolor='black', linewidth=0.5, binwidth=0.2)
    # axes[1].set_title(r"Distribution: $\mathrm{Log}_2$ Transformed") # Commented for Overleaf
    axes[1].set_xlabel(r"$\log_2$ Ratio")

    # Z-scored Plot
    sns.histplot(z_vals, kde=False, ax=axes[2], color='#154734',
                 edgecolor='black', linewidth=0.5, binwidth=0.2)
    # axes[2].set_title("Distribution: Z-scored") # Commented for Overleaf
    axes[2].set_xlabel("Z-score")

    # Formatting
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_name, format='pdf', bbox_inches='tight')
    plt.close()

def plot_triple_clustered_heatmap(df, ratio_cols, output_name="clustered_heatmaps.pdf"):
    # Prepare data for Graph 1 & 2 (Original Order)
    log_cols = [f"Log2_{c}" for c in ratio_cols]
    X_raw_orig = df[ratio_cols].values
    X_log_orig = df[log_cols].values

    # Calculate Z-score for original order
    mean_orig = np.nanmean(X_log_orig, axis=1, keepdims=True)
    std_orig = np.nanstd(X_log_orig, axis=1, keepdims=True)
    X_z_orig = (X_log_orig - mean_orig) / np.where(std_orig == 0, 1.0, std_orig)

    # Prepare data for Graph 3 (Clustered Order)
    df_sorted = df.sort_values("Cluster")
    X_log_clust = df_sorted[log_cols].values

    # Calculate Z-score for clustered order
    mean_clust = np.nanmean(X_log_clust, axis=1, keepdims=True)
    std_clust = np.nanstd(X_log_clust, axis=1, keepdims=True)
    X_z_clust = (X_log_clust - mean_clust) / np.where(std_clust == 0, 1.0, std_clust)

    fig, axes = plt.subplots(1, 3, figsize=(20, 10))

    # Graph 1: Raw Ratios with Logarithmic Scaling
    sns.heatmap(
        X_raw_orig,
        ax=axes[0],
        cmap="RdBu_r",
        norm=LogNorm(vmin=0.1, vmax=10),
        yticklabels=False
    )
    axes[0].set_title("Raw Ratios (Log-Scaled Color)")

    # Graph 2: Z-scored (Original Order)
    sns.heatmap(
        X_z_orig,
        ax=axes[1],
        cmap="RdBu_r",
        center=0,
        yticklabels=False
    )
    axes[1].set_title("Z-scored Log2 (Original)")

    # Graph 3: Z-scored (Clustered Order)
    sns.heatmap(
        X_z_clust,
        ax=axes[2],
        cmap="RdBu_r",
        center=0,
        yticklabels=False
    )
    axes[2].set_title("Z-scored Log2 (Clustered)")

    # Standardize x-axis formatting for all plots
    for ax in axes:
        ax.set_xticklabels(ratio_cols, rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_name, format="pdf", bbox_inches="tight")
    plt.close()

def zscore_per_gene_df(df: pd.DataFrame, cols: list[str], prefix: str = "Z_") -> pd.DataFrame:
    X = df[cols].to_numpy(dtype=np.float64)

    mean = np.nanmean(X, axis=1, keepdims=True)
    std = np.nanstd(X, axis=1, keepdims=True)
    std[std == 0] = 1.0  # constant genes -> all zeros after centering

    Z = (X - mean) / std

    return pd.DataFrame(
        Z,
        columns=[f"{prefix}{c}" for c in cols],
        index=df.index
    )


def plot_silhouette_comparison(hier_results, kmeans_results):
    """
    Plot silhouette score vs k for hierarchical clustering and k-means.
    Annotates points with cluster sizes in a clean [num1, num2] format.
    """
    ks = sorted(hier_results.keys())

    hier_scores = [hier_results[k]["silhouette"] for k in ks]
    kmeans_scores = [kmeans_results[k]["silhouette"] for k in ks]

    plt.figure(figsize=(10, 6))
    plt.plot(
        ks,
        hier_scores,
        marker="o",
        color="#c31e23",
        label="Hierarchical (Pearson correlation)"
    )
    plt.plot(
        ks,
        kmeans_scores,
        marker="o",
        color="#0d7d87",
        label="k-means (Euclidean)"
    )

    # Annotate each point with cluster sizes
    for k in ks:
        # Hierarchical Labels
        h_labels = hier_results[k]["labels"]
        _, h_counts = np.unique(h_labels, return_counts=True)
        # .tolist() removes the np.int64 wrapper for a clean string
        h_size_str = f"sizes: {h_counts.tolist()}"

        plt.annotate(
            h_size_str,
            xy=(k, hier_results[k]["silhouette"]),
            xytext=(10, 10),
            textcoords="offset points",
            ha="left",
            fontsize=8,
            color="#000000",
            fontweight='bold'
        )

        # K-means Labels
        km_labels = kmeans_results[k]["labels"]
        _, km_counts = np.unique(km_labels, return_counts=True)
        km_size_str = f"sizes: {km_counts.tolist()}"

        offset = 6 if k == 4 else 0

        plt.annotate(
            km_size_str,
            xy=(k, kmeans_results[k]["silhouette"]),
            xytext=(10, 10-offset),
            textcoords="offset points",
            ha="left",
            fontsize=8,
            color="#000000",
            fontweight='bold'
        )
    # ----------------------------------------------

    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    #plt.title("Silhouette Score vs k (Annotated with Cluster Sizes)")

    plt.legend(loc='upper right')

    ax = plt.gca()
    ax.set_xticks(ks)

    # Padding to ensure labels near the top/right edges aren't cut off
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min - 0.05, y_max + 0.1)
    ax.set_xlim(min(ks) - 0.5, max(ks) + 1.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("silhouette_comparison.pdf", format="pdf", dpi=600, bbox_inches="tight")
    plt.close()


def plot_method_vs_author_silhouette(m1_results, author_results, output_name="method_vs_author_silhouette.pdf"):
    """
    Compares silhouette scores for two different gene subsets using hierarchical clustering.
    """
    ks = sorted(m1_results.keys())
    m1_scores = [m1_results[k]["silhouette"] for k in ks]
    auth_scores = [author_results[k]["silhouette"] for k in ks]

    plt.figure(figsize=(10, 6))
    plt.plot(ks, m1_scores, marker="o", color="#c31e23", label="Method 1 (Top 230)")
    plt.plot(ks, auth_scores, marker="o", color="#0d7d87", label="Authors (Top 230)")

    for k in ks:
        # Annotation for Method 1
        m1_counts = np.unique(m1_results[k]["labels"], return_counts=True)[1]
        plt.annotate(f"sizes: {m1_counts.tolist()}", xy=(k, m1_results[k]["silhouette"]),
                     xytext=(10, 10), textcoords="offset points", fontsize=8, fontweight='bold')

        # Annotation for Authors
        auth_counts = np.unique(author_results[k]["labels"], return_counts=True)[1]
        plt.annotate(f"sizes: {auth_counts.tolist()}", xy=(k, author_results[k]["silhouette"]),
                     xytext=(10, -15), textcoords="offset points", fontsize=8, fontweight='bold')

    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.legend(loc='upper right')

    ax = plt.gca()
    ax.set_xticks(ks)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_name, format="pdf", dpi=600)
    plt.close()

def plot_selection_comparison_heatmap(df_m1, df_author, ratio_cols, output_name="method_vs_author_heatmap.pdf"):
    """
    Plots side-by-side clustered heatmaps for Method 1 and Author gene selections.
    Expects both DataFrames to have a 'Cluster' column.
    """
    log_cols = [f"Log2_{c}" for c in ratio_cols]

    # Sort both by cluster to show grouped patterns
    m1_sorted = df_m1.sort_values("Cluster")
    auth_sorted = df_author.sort_values("Cluster")

    # Helper to get Z-scored data for plotting
    def get_z_data(df):
        X = df[log_cols].values
        mean = np.nanmean(X, axis=1, keepdims=True)
        std = np.nanstd(X, axis=1, keepdims=True)
        return (X - mean) / np.where(std == 0, 1.0, std)

    X_m1_z = get_z_data(m1_sorted)
    X_auth_z = get_z_data(auth_sorted)

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    # Graph 1: Method 1 Clustered
    sns.heatmap(
        X_m1_z,
        ax=axes[0],
        cmap="RdBu_r",
        center=0,
        yticklabels=False,
        cbar_kws={'label': 'Z-score'}
    )
    axes[0].set_title(f"Method 1: Top 230 (n={len(df_m1)})")

    # Graph 2: Authors Clustered
    sns.heatmap(
        X_auth_z,
        ax=axes[1],
        cmap="RdBu_r",
        center=0,
        yticklabels=False,
        cbar_kws={'label': 'Z-score'}
    )
    axes[1].set_title(f"Authors: Top 230 (n={len(df_author)})")

    # Formatting
    for ax in axes:
        ax.set_xticklabels(ratio_cols, rotation=45, ha="right")
        ax.set_xlabel("Time Points")

    plt.tight_layout()
    plt.savefig(output_name, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Comparison heatmap saved to {output_name}")
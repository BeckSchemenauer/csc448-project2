import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from matplotlib.colors import LogNorm


def load_and_transform(file_path: str = "Data/diauxic_raw_ratios.txt"):
    df = pd.read_csv(file_path, sep="\t")
    ratio_cols = [col for col in df.columns if 'Ratio' in col]

    log_df = df.copy()
    for col in ratio_cols:
        log_df[f"Log2_{col}"] = np.log2(df[col].replace(0, np.nan))

    return df, log_df, ratio_cols


def visualize_to_pdf(df_raw, df_log, ratio_cols, output_name="distribution_plots.pdf"):
    sns.set_context("talk")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Flatten and clean data
    raw_vals = df_raw[ratio_cols].values.flatten()
    log_vals = df_log[[f"Log2_{c}" for c in ratio_cols]].values.flatten()
    raw_vals = raw_vals[~np.isnan(raw_vals)]
    log_vals = log_vals[~np.isnan(log_vals)]

    # Raw Ratios Plot - added edgecolor and linewidth
    sns.histplot(raw_vals, kde=False, ax=axes[0], color='#154734',
                 edgecolor='black', linewidth=0.5, binwidth=0.1)
    axes[0].set_title("Distribution: Raw Ratios")
    axes[0].set_xlabel("Ratio Value")

    # Log2 Plot with subscript in title - added edgecolor and linewidth
    sns.histplot(log_vals, kde=False, ax=axes[1], color='#154734',
                 edgecolor='black', linewidth=0.5, binwidth=0.2)
    axes[1].set_title(r"Distribution: $\mathrm{Log}_2$ Transformed")
    axes[1].set_xlabel(r"$\log_2$ Ratio")

    # Decrease tick mark font size (labelsize)
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.savefig(output_name, format='pdf', bbox_inches='tight')
    plt.close()

def plot_triple_clustered_heatmap(df, ratio_cols, output_name="clustered_heatmaps.pdf"):
    # 1. Prepare data for Graph 1 & 2 (Original Order)
    log_cols = [f"Log2_{c}" for c in ratio_cols]
    X_raw_orig = df[ratio_cols].values
    X_log_orig = df[log_cols].values

    # Calculate Z-score for original order
    mean_orig = np.nanmean(X_log_orig, axis=1, keepdims=True)
    std_orig = np.nanstd(X_log_orig, axis=1, keepdims=True)
    X_z_orig = (X_log_orig - mean_orig) / np.where(std_orig == 0, 1.0, std_orig)

    # 2. Prepare data for Graph 3 (Clustered Order)
    df_sorted = df.sort_values("Cluster")
    X_log_clust = df_sorted[log_cols].values

    # Calculate Z-score for clustered order
    mean_clust = np.nanmean(X_log_clust, axis=1, keepdims=True)
    std_clust = np.nanstd(X_log_clust, axis=1, keepdims=True)
    X_z_clust = (X_log_clust - mean_clust) / np.where(std_clust == 0, 1.0, std_clust)

    fig, axes = plt.subplots(1, 3, figsize=(20, 10))

    # Graph 1: Raw Ratios with Logarithmic Scaling
    # center=1.0 is the "no change" point for ratios
    sns.heatmap(
        X_raw_orig,
        ax=axes[0],
        cmap="RdBu_r",
        norm=LogNorm(vmin=0.1, vmax=10), # Adjust vmin/vmax based on your data spread
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
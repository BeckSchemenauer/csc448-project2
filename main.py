from eda import (
    load_and_transform,
    plot_distribution,
    zscore_per_gene_df,
    plot_triple_clustered_heatmap,
    plot_silhouette_comparison,
    plot_method_vs_author_silhouette,
    plot_selection_comparison_heatmap,
)
from clustering import (
    correlation_distance_condensed,
    hierarchical_silhouette_by_k,
    kmeans_silhouette_by_k,
)
from gene_selection import (
    extract_author_230,
    select_top_genes_min_max,
    select_genes_by_mean_shift,
)
import numpy as np

# ----------------------------------
# Data setup
# ----------------------------------
print("Loading and transforming data...")
raw_df, processed_df, ratio_cols = load_and_transform()

# Prepare Z-scored data (used for both methods now)
log_cols = [f"Log2_{c}" for c in ratio_cols]
log_zscore_df = zscore_per_gene_df(processed_df, log_cols)
X_log_z = log_zscore_df.to_numpy(dtype=np.float64)
X_log_z = np.nan_to_num(X_log_z, nan=0.0)
print("Finished loading and transforming data...")

print("Computing distances for hierarchical clustering...")
# Hierarchical: Compute condensed correlation-distance vector
# Using X_log_z here produces the same result as X_log
dist_vec = correlation_distance_condensed(X_log_z)

# ----------------------------------
# Clustering
# ----------------------------------

print("Hierarchical Clustering...")
hier_results = hierarchical_silhouette_by_k(
    dist_vec,
    k_range=range(2, 6),
    method="average"
)

print("K-Means Clustering...")
kmeans_results = kmeans_silhouette_by_k(
    X_log_z,
    k_range=range(2, 6),
    random_state=42,
    n_init=20
)

# ----------------------------------
# Plotting and Evaluation
# ----------------------------------
plot_silhouette_comparison(hier_results, kmeans_results)

# Plot Heatmaps
k_choice = 3
processed_df['Cluster'] = hier_results[k_choice]["labels"]
plot_triple_clustered_heatmap(processed_df, ratio_cols)

# Distribution plots
print("Generating distribution plots...")
plot_distribution(raw_df, processed_df, log_zscore_df, ratio_cols)
print("Done.")

# ----------------------------------
# Get top genes
# ----------------------------------
author_genes = extract_author_230(
    input_file="data/230_authors.txt",
    output_file="top_230_author.txt"
)

# Select top 230 genes by metric 1
top_genes = select_top_genes_min_max(df=processed_df, log_cols=log_cols, n_genes=230,
                                     output_file="top_230_method1.txt")

# Select top 230 genes by metric 2
top_genes_method2 = select_genes_by_mean_shift(
    df=processed_df,
    log_cols=log_cols,
    top_gain_n=115,
    top_drop_n=115,
    output_file="top_230_method2.txt",
)

# ----------------------------------
# Analysis of Method 1 Top 230 Genes
# ----------------------------------

# Extract the gene names from Method 1 results
m1_gene_names = [item[0] for item in top_genes]
m1_subset_df = processed_df[processed_df['ORF'].isin(m1_gene_names)].copy()
X_m1 = log_zscore_df.loc[m1_subset_df.index].to_numpy(dtype=np.float64)
X_m1 = np.nan_to_num(X_m1, nan=0.0)

# Compute Distance and Clustering
dist_vec_m1 = correlation_distance_condensed(X_m1)
m1_clustering_results = hierarchical_silhouette_by_k(
    dist_vec_m1,
    k_range=range(2, 6),
    method="average"
)

# ----------------------------------
# Analysis of Method 1 vs Authors Top 230
# ----------------------------------

# 1. Process Authors' Genes
# Filter processed_df to only include the 230 genes specified by authors
author_subset_df = processed_df[processed_df['ORF'].isin(author_genes)].copy()
X_author = log_zscore_df.loc[author_subset_df.index].to_numpy(dtype=np.float64)
X_author = np.nan_to_num(X_author, nan=0.0)

# 2. Compute Clustering for Authors
print("Clustering Author gene set...")
dist_vec_author = correlation_distance_condensed(X_author)
author_clustering_results = hierarchical_silhouette_by_k(
    dist_vec_author,
    k_range=range(2, 6),
    method="average"
)

print("\n" + "=" * 40)
print("CLUSTERING COMPARISON: METHOD 1 VS AUTHORS")
print("=" * 40)

for k in range(2, 6):
    m1_score = m1_clustering_results[k]["silhouette"]
    auth_score = author_clustering_results[k]["silhouette"]

    print(f"k={k}:")
    print(f"  Method 1 Silhouette: {m1_score:.4f}")
    print(f"  Authors  Silhouette: {auth_score:.4f}")
    print(f"  Difference: {m1_score - auth_score:+.4f}")
    print("-" * 20)

# 3. Use the updated function to compare Method 1 and Author Results
print("Generating comparison plot...")
# Ensure you import this new function at the top of main.py or define it locally
plot_method_vs_author_silhouette(m1_clustering_results, author_clustering_results)

# 4. Final results dictionary for inspection
analysis_results = {
    "method_1": m1_clustering_results,
    "authors": author_clustering_results
}

print("Plot saved as 'method_vs_author_silhouette.pdf'")

# Assign clusters
k_choice = 2
m1_subset_df['Cluster'] = m1_clustering_results[k_choice]["labels"]
author_subset_df['Cluster'] = author_clustering_results[k_choice]["labels"]

# Plot the side-by-side comparison
plot_selection_comparison_heatmap(
    m1_subset_df,
    author_subset_df,
    ratio_cols
)
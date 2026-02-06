from eda import (
    load_and_transform,
    visualize_to_pdf,
    zscore_per_gene_df,
    plot_triple_clustered_heatmap,
    plot_silhouette_comparison
)
from clustering import (
    correlation_distance_condensed,
    hierarchical_silhouette_by_k,
    kmeans_silhouette_by_k,
)
import numpy as np

# 1) Load + transform
print("Loading and transforming data...")
raw_df, processed_df, ratio_cols = load_and_transform()
print("Finished loading and transforming data...")

# 2) Build log2 expression matrix
X_log = processed_df[[f"Log2_{c}" for c in ratio_cols]].to_numpy(dtype=np.float64)
X_log = np.nan_to_num(X_log, nan=0.0)  # Handle NaNs from log2(0)

# 3) Compute condensed correlation-distance vector (upper triangle)
dist_vec = correlation_distance_condensed(X_log)

hier_results = hierarchical_silhouette_by_k(
    dist_vec,
    k_range=range(2, 6),
    method="average"
)

# 5) Z-score log2 data (per gene) for k-means
log_cols = [f"Log2_{c}" for c in ratio_cols]
log_zscore_df = zscore_per_gene_df(processed_df, log_cols)
X_log_z = log_zscore_df.to_numpy(dtype=np.float64)
X_log_z = np.nan_to_num(X_log_z, nan=0.0)

# 6) k-means: silhouette-by-k (k=2..5)
kmeans_results = kmeans_silhouette_by_k(
    X_log_z,
    k_range=range(2, 6),
    random_state=42,
    n_init=20
)

# 7) Plot both methods together
plot_silhouette_comparison(hier_results, kmeans_results)

# (optional) print the scores
print("\nHierarchical silhouette:")
for k in sorted(hier_results):
    print(f"{k}: {hier_results[k]['silhouette']}")

print("\nk-means silhouette:")
for k in sorted(kmeans_results):
    print(f"{k}: {kmeans_results[k]['silhouette']}")

# plot heatmap comparison
k_choice = 3
hier_labels = hier_results[k_choice]["labels"]

# Add labels to your dataframe to facilitate sorting
processed_df['Cluster'] = hier_labels

# The plotting function will use the 'Cluster' column to sort for the 3rd plot.
plot_triple_clustered_heatmap(processed_df, ratio_cols)

print("Generating distribution plots...")
visualize_to_pdf(raw_df, processed_df, log_zscore_df, ratio_cols)
print("Done.")
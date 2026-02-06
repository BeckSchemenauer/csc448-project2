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

# 2) Prepare Z-scored data (used for both methods now)
log_cols = [f"Log2_{c}" for c in ratio_cols]
log_zscore_df = zscore_per_gene_df(processed_df, log_cols)
X_log_z = log_zscore_df.to_numpy(dtype=np.float64)
X_log_z = np.nan_to_num(X_log_z, nan=0.0)

# 3) Hierarchical: Compute condensed correlation-distance vector
# Using X_log_z here produces the same result as X_log
dist_vec = correlation_distance_condensed(X_log_z)

hier_results = hierarchical_silhouette_by_k(
    dist_vec,
    k_range=range(2, 6),
    method="average"
)

# 4) k-means: silhouette-by-k (k=2..5)
kmeans_results = kmeans_silhouette_by_k(
    X_log_z,
    k_range=range(2, 6),
    random_state=42,
    n_init=20
)

# 5) Plotting and Evaluation
plot_silhouette_comparison(hier_results, kmeans_results)

# 6) Plot Heatmaps
k_choice = 3
processed_df['Cluster'] = hier_results[k_choice]["labels"]
plot_triple_clustered_heatmap(processed_df, ratio_cols)

# 7) Distribution plots
print("Generating distribution plots...")
visualize_to_pdf(raw_df, processed_df, log_zscore_df, ratio_cols)
print("Done.")
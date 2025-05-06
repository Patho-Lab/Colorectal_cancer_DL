# --- Refined R code for Elbow Plot (SSE) for UMAP-based Hierarchical Clustering ---

# ------------------- 1. Load Data and Perform PCA (Keep as is) -------------------
# 1. Load the .csv file
csv_file_path <- "" # Replace with the actual path to your .csv file
if (!file.exists(csv_file_path)) {
  stop(paste("CSV file not found:", csv_file_path))
}
feature_df <- read.csv(csv_file_path)

# 2. Prepare Data for PCA
feature_columns <- feature_df[, -1]
feature_matrix <- as.matrix(feature_columns)
scaled_feature_matrix <- scale(feature_matrix, center = TRUE, scale = TRUE)

# 3. Perform PCA
pca_result <- prcomp(scaled_feature_matrix, center = FALSE, scale. = FALSE)
pc_scores <- pca_result$x


# ------------------- 2. Perform UMAP Embedding -------------------
# 1. Load UMAP Library
if (!requireNamespace("umap", quietly = TRUE)) {
  install.packages("umap-learn") # Suggest installation if not found
}
library(umap)

# 2. Perform UMAP on PCA Scores (for clustering evaluation)
umap_result <- umap(pc_scores) # Default n_components = 2 (can be adjusted if needed for evaluation)
umap_scores <- umap_result$layout


# ------------------- 3. Perform Hierarchical Clustering on UMAP Embeddings -------------------
# 1. Load Necessary Libraries
library(FNN) # (Likely not needed for evaluation itself, but kept if you use other parts of the script)
library(stats)
library(ggplot2)

# 2. Calculate Distance Matrix on UMAP scores
distance_matrix_umap <- dist(umap_scores, method = "euclidean")

# 3. Perform Hierarchical Clustering on UMAP Distance Matrix
hierarchical_clustering_result_umap <- hclust(distance_matrix_umap, method = "ward.D2")


# ------------------- 3.1. Elbow Plot for Choosing k (UMAP-based) -------------------
# Initialize vectors to store SSE values for different k's
max_k <- 30 # Elongate x-axis to k=30 (adjust as needed)
sse_values_umap <- numeric(max_k)

# Loop through different values of k
for (k_val in 1:max_k) {
  # Cut the dendrogram to get clusters for the current k (UMAP-based clustering)
  cluster_assignments_eval_umap <- cutree(hierarchical_clustering_result_umap, k = k_val)
  
  # --- Calculate SSE (using UMAP scores) ---
  current_sse_umap <- 0
  for (cluster_num in 1:k_val) {
    # Get UMAP data points in the current cluster
    cluster_points_umap <- umap_scores[cluster_assignments_eval_umap == cluster_num, ]
    
    # If the cluster is not empty
    if(nrow(cluster_points_umap) > 0) {
      # Calculate the centroid of the cluster in UMAP space
      centroid_umap <- colMeans(cluster_points_umap)
      
      # Calculate SSE for points in this cluster (using UMAP coordinates)
      for (i in 1:nrow(cluster_points_umap)) {
        current_sse_umap <- current_sse_umap + sum((cluster_points_umap[i, ] - centroid_umap)^2)
      }
    }
  }
  sse_values_umap[k_val] <- current_sse_umap
}


# Create a data frame for plotting SSE
eval_data_umap <- data.frame(k = 1:max_k,
                             SSE = sse_values_umap)

# Generate the Elbow Plot for UMAP-based Clustering
elbow_plot_umap <- ggplot(eval_data_umap, aes(x = k, y = SSE)) +
  geom_line(color = "blue", linetype = "solid") +
  geom_point(color = "blue") +
  scale_x_continuous(breaks = 1:max_k) +
  labs(title = "Elbow Plot (SSE) for UMAP-based Hierarchical Clustering", # Updated title
       x = "Number of Clusters (k)",
       y = "Sum of Squared Errors (SSE)") +
  theme_minimal()

print(elbow_plot_umap)


# --- 3.2. (Rest of the code removed for Elbow Plot Only) ---
# --- 3. PCA Plot (Removed) ---
# --- 4. UMAP Plot (Removed) ---
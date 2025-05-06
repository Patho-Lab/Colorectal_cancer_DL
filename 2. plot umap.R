# --- Refined R code for PCA and UMAP plots colored by UMAP-based Hierarchical Clusters ---

# ------------------- 1. Load Data and Perform PCA (Keep as is) -------------------
# 1. Load the .csv file
csv_file_path <- "" # Replace with the actual path to your .csv file
if (!file.exists(csv_file_path)) {
  stop(paste("CSV file not found:", csv_file_path))
}
feature_df <- read.csv(csv_file_path)

# 2. Prepare Data for PCA
filenames <- feature_df[, 1] # Extract filenames from the first column
feature_columns <- feature_df[, -1]
feature_matrix <- as.matrix(feature_columns)
scaled_feature_matrix <- scale(feature_matrix, center = TRUE, scale = TRUE)

# 3. Perform PCA
pca_result <- prcomp(scaled_feature_matrix, center = FALSE, scale. = FALSE)
pc_scores <- pca_result$x


# ------------------- 2. Perform UMAP Embedding (Keep as is) -------------------
# 1. Load UMAP Library
if (!requireNamespace("umap", quietly = TRUE)) {
  install.packages("umap-learn") # Suggest installation if not found
}
library(umap)

# 2. Perform UMAP on PCA Scores (for 2D plot)
umap_result_2d <- umap(pc_scores) # Default n_components = 2
umap_scores_2d <- umap_result_2d$layout


# ------------------- 3. Perform Hierarchical Clustering on UMAP Embeddings (MODIFIED) -------------------
# 1. Load Necessary Libraries (already loaded above, no need to reload)
library(FNN)
library(stats)
library(ggplot2)

# 2. Use UMAP Scores for Clustering (MODIFIED - using umap_scores_2d instead of pc_scores)

# 3. Calculate Distance Matrix on UMAP scores (MODIFIED)
distance_matrix_umap <- dist(umap_scores_2d, method = "euclidean") # Distance matrix on UMAP

# 4. Perform Hierarchical Clustering on UMAP Distance Matrix (MODIFIED)
hierarchical_clustering_result_umap <- hclust(distance_matrix_umap, method = "ward.D2")

# 5. Cut the Dendrogram to Get Clusters (MODIFIED - using UMAP clustering result)
k <- 5 # You can experiment with different values of k
cluster_assignments_umap <- cutree(hierarchical_clustering_result_umap, k = k)

# --- Define a more distinct color palette ---
distinct_colors <- c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", "#A65628", "#F781BF", "#999999", "#000000") # Example palette, can be customized
if(k > length(distinct_colors)){
  warning("Number of clusters k is larger than the number of distinct colors defined in 'distinct_colors'. Colors may be reused.")
}
plot_palette <- distinct_colors[1:k] # Use only as many colors as needed for k clusters


# ------------------- 4. PCA Plot Colored by UMAP-based Hierarchical Clusters (MODIFIED) -------------------
# --- Visualize Clusters in 2D PCA Space (PC1 vs PC2) ---
# Create a data frame for plotting
plot_data_pca_umap_clusters <- data.frame(PC1 = pc_scores[, 1],
                                          PC2 = pc_scores[, 2],
                                          Cluster = factor(cluster_assignments_umap)) # Factor for colors - UMAP Clusters

if(ncol(pc_scores) >= 2){ # Check if there are at least 2 PCs to plot
  cluster_plot_pca_umap_clusters <- ggplot(plot_data_pca_umap_clusters, aes(x = PC1, y = PC2, color = Cluster)) +
    geom_point() +
    labs(title = paste("PCA (PC1 vs PC2) Colored by UMAP-based Hierarchical Clusters (k =", k, ")"), # MODIFIED Title
         x = "Principal Component 1",
         y = "Principal Component 2",
         color = "UMAP-based Cluster") + # MODIFIED Legend Title
    theme_minimal() + # Keep theme_minimal and add:
    theme(panel.background = element_blank(),  # Remove panel background
          panel.grid.major = element_blank(),   # Remove major grid lines
          panel.grid.minor = element_blank(),    # Remove minor grid lines
          axis.line = element_line(colour = "black"), # Ensure axis lines are visible in black
          legend.position = c(0.85, 0.85) # Position legend inside the plot (top-right corner, adjust as needed)
    ) +
    scale_color_manual(values = plot_palette) # Apply the distinct color palette
  
  print(cluster_plot_pca_umap_clusters)
} else {
  print("PCA cluster plot not generated: PCA result has less than 2 principal components.")
}


# ------------------- 5. 2D UMAP Plot Colored by UMAP-based Hierarchical Clusters (MODIFIED) -------------------
# (Using the same 2D UMAP embeddings as before, but now coloring by UMAP-based clusters)

# 3. Create Data Frame for UMAP Plotting (2D)
plot_data_umap_umap_clusters_2d <- data.frame(UMAP1 = umap_scores_2d[, 1],
                                              UMAP2 = umap_scores_2d[, 2],
                                              Cluster = factor(cluster_assignments_umap)) # Factor for colors - UMAP Clusters

# 4. Generate 2D UMAP Plot
if(ncol(umap_scores_2d) >= 2){ # Check if there are at least 2 UMAP dimensions to plot
  umap_cluster_plot_umap_clusters_2d <- ggplot(plot_data_umap_umap_clusters_2d, aes(x = UMAP1, y = UMAP2, color = Cluster)) +
    geom_point() +
    labs(title = paste(""), # MODIFIED Title
         x = "UMAP Dimension 1",
         y = "UMAP Dimension 2",
         color = "Cluster") + # MODIFIED Legend Title
    theme_minimal() + # Keep theme_minimal and add:
    theme(panel.background = element_blank(),  # Remove panel background
          panel.grid.major = element_blank(),   # Remove major grid lines
          panel.grid.minor = element_blank(),    # Remove minor grid lines
          axis.line = element_line(colour = "black"), # Ensure axis lines are visible in black
          legend.position = c(0.85, 0.15) # Position legend inside the plot (top-right corner, adjust as needed)
    ) +
    scale_color_manual(values = plot_palette) # Apply the distinct color palette
  
  print(umap_cluster_plot_umap_clusters_2d)
} else {
  print("2D UMAP cluster plot not generated: UMAP result has less than 2 dimensions.")
}



# ------------------- 7. Record Filename and UMAP-based Cluster to CSV (MODIFIED) -------------------
# Create a data frame for filename and cluster
filename_cluster_df_umap <- data.frame(filename = filenames,
                                       umap_hierarchical_cluster = cluster_assignments_umap) # MODIFIED column name

# Define the output CSV file name
output_csv_filename_umap <- paste0(tools::file_path_sans_ext(basename(csv_file_path)), "_umap_cluster_assignments.csv") # MODIFIED filename
output_csv_path_umap <- file.path(dirname(csv_file_path), output_csv_filename_umap) # Save in the same directory as input

# Write the data frame to a CSV file
write.csv(filename_cluster_df_umap, file = output_csv_path_umap, row.names = FALSE)

cat(paste("Filename and UMAP-based Hierarchical Cluster assignments saved to:", output_csv_path_umap, "\n"))
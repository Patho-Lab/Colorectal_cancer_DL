# --- R code to Arrange Images Based on UMAP Hierarchical Clusters ---

# 1. Load UMAP Cluster Assignment CSV File (MODIFIED for UMAP clusters)
cluster_assignment_csv_path <- "" # Replace with the actual path to your UMAP cluster assignment CSV file
if (!file.exists(cluster_assignment_csv_path)) {
  stop(paste("UMAP Cluster assignment CSV file not found:", cluster_assignment_csv_path))
}
cluster_df <- read.csv(cluster_assignment_csv_path)

# 2. Define Image Source Directory (Keep as is)
image_source_dir <- "" # Replace with the actual path to the directory containing your images
if (!dir.exists(image_source_dir)) {
  stop(paste("Image source directory not found:", image_source_dir))
}

# 3. Define Output Base Directory for Clustered Images (Keep as is or modify if desired)
output_base_dir <- "" # Modified output base directory to indicate UMAP clusters (optional)
if (!dir.exists(output_base_dir)) {
  dir.create(output_base_dir, recursive = TRUE) # Create output base directory if it doesn't exist
}

# 4. Create UMAP Cluster Directories and Copy Images (MODIFIED for UMAP clusters)
unique_clusters <- unique(cluster_df$umap_hierarchical_cluster) # Use umap_hierarchical_cluster column

for (cluster_num in unique_clusters) {
  # a. Create Cluster Directory (MODIFIED directory name to indicate UMAP clusters - optional)
  cluster_dir_name <- paste0("UMAP_Cluster_", cluster_num) # Modified cluster directory name
  cluster_output_dir <- file.path(output_base_dir, cluster_dir_name)
  
  if (!dir.exists(cluster_output_dir)) {
    dir.create(cluster_output_dir, recursive = TRUE) # Create cluster directory if it doesn't exist
  }
  
  # b. Get Filenames for the Current Cluster (MODIFIED to use umap_hierarchical_cluster column)
  filenames_in_cluster <- cluster_df$filename[cluster_df$umap_hierarchical_cluster == cluster_num] # Use umap_hierarchical_cluster column
  
  # c. Copy Images to Cluster Directory (Keep as is)
  for (filename in filenames_in_cluster) {
    # Construct Source and Destination Paths
    source_image_path <- file.path(image_source_dir, filename) # Assuming filename in CSV is just image name
    destination_image_path <- file.path(cluster_output_dir, filename)
    
    # Check if source image exists before copying
    if (file.exists(source_image_path)) {
      file.copy(from = source_image_path, to = destination_image_path)
      cat(paste("Copied:", filename, "to", cluster_dir_name, "\n"))
    } else {
      cat(paste("Warning: Source image not found:", source_image_path, "\n"))
    }
  }
  cat(paste("Images for", cluster_dir_name, "arranged successfully.\n"))
}

cat("Image arrangement based on UMAP hierarchical clusters completed.\n")
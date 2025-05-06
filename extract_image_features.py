#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extracts features from a directory of PNG images using a specified pre-trained 
encoder model (e.g., Conch).

Leverages GPU if available for faster processing. Saves the extracted features 
along with their corresponding filenames into a single CSV file. Each feature 
dimension is saved as a separate column.
"""

import torch
from models import get_encoder # Assuming 'models.py' with 'get_encoder' is in the same directory or PYTHONPATH
from PIL import Image
import os
import pandas as pd
import warnings
from typing import Optional # For type hinting
from tqdm import tqdm # For progress bar

# --- Configuration ---
# Model and Image Settings
MODEL_NAME = 'conch_v1_5'    # <-- **REPLACE with your desired model name (e.g., 'conch_v1_5', 'resnet50')**
TARGET_IMG_SIZE = 448      # <-- **REPLACE with the input size expected by the model**

# Input/Output Paths
INPUT_DIR = '' # <-- **REPLACE with the path to your directory of PNG images**
OUTPUT_SUBDIR = './extracted_features' # Subdirectory to save the output CSV
OUTPUT_CSV_FILENAME = 'directory_features_with_filenames.csv' # Name for the output CSV file
# --- End Configuration ---


def setup_feature_extractor(model_name: str, img_size: int, device: torch.device):
    """Loads the feature extractor model and image transformations."""
    print(f"Setting up feature extractor: {model_name} with image size {img_size}...")
    try:
        feature_extractor, img_transforms = get_encoder(model_name, img_size)
        feature_extractor.eval() # Set to evaluation mode
        feature_extractor.to(device) # Move model to the specified device
        print(f"Model '{model_name}' loaded successfully and moved to {device}.")
        return feature_extractor, img_transforms
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")
        print("Please ensure the model name is correct and the 'models.py' script is accessible.")
        return None, None


def extract_features_from_png(image_path: str, 
                              model: torch.nn.Module, 
                              transforms: callable, 
                              device: torch.device) -> Optional[torch.Tensor]:
    """
    Extracts features from a single PNG image.

    Args:
        image_path (str): Path to the PNG image file.
        model (torch.nn.Module): The loaded feature extractor model.
        transforms (callable): The image transformations function.
        device (torch.device): The device (CPU or CUDA) to use for processing.

    Returns:
        Optional[torch.Tensor]: Extracted features as a torch Tensor (on CPU).
                                Returns None if there's an error.
    """
    try:
        img = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error loading image '{os.path.basename(image_path)}': {e}")
        return None

    try:
        img_tensor = transforms(img)
        # Add batch dimension [C, H, W] -> [1, C, H, W] and move to device
        img_tensor = img_tensor.unsqueeze(0).to(device) 
    except Exception as e:
        print(f"Error transforming image '{os.path.basename(image_path)}': {e}")
        return None

    # Perform inference without calculating gradients
    with torch.no_grad():
        try:
            features = model(img_tensor)
            # Move features to CPU before returning for easier handling (e.g., numpy conversion)
            return features.cpu() 
        except Exception as e:
            print(f"Error during feature extraction for '{os.path.basename(image_path)}': {e}")
            return None


def main():
    """Main function to orchestrate feature extraction for a directory."""

    # 1. Define Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")
        warnings.warn("CUDA not available, running on CPU. Feature extraction will be slower.")

    # 2. Setup Feature Extractor
    feature_extractor, img_transforms = setup_feature_extractor(MODEL_NAME, TARGET_IMG_SIZE, device)
    if feature_extractor is None:
        return # Exit if model setup failed

    # 3. Validate Input Directory
    if not os.path.isdir(INPUT_DIR):
        print(f"Error: Input directory not found at '{INPUT_DIR}'. Please provide a valid directory path.")
        return

    # 4. Find Image Files
    try:
        all_files = os.listdir(INPUT_DIR)
        image_files = [f for f in all_files if f.lower().endswith('.png')]
    except OSError as e:
        print(f"Error reading directory '{INPUT_DIR}': {e}")
        return
        
    if not image_files:
        print(f"No PNG images found in directory: '{INPUT_DIR}'")
        return
    
    print(f"Found {len(image_files)} PNG images in directory: '{INPUT_DIR}'")

    # 5. Process Images and Extract Features
    data_rows = [] # List to store data for the final DataFrame
    feature_length = None # To store the length of the first successful feature vector

    print("Starting feature extraction...")
    # Use tqdm for a progress bar
    for image_file in tqdm(image_files, desc="Extracting Features"):
        image_file_path = os.path.join(INPUT_DIR, image_file)
        
        # Extract features (function returns features on CPU)
        extracted_features_tensor = extract_features_from_png(image_file_path, feature_extractor, img_transforms, device)

        if extracted_features_tensor is not None:
            # Flatten features to 1D numpy array, then list
            feature_list = extracted_features_tensor.numpy().flatten().tolist() 
            
            # Check consistency of feature length
            if feature_length is None:
                feature_length = len(feature_list)
                print(f"Detected feature vector length: {feature_length}")
            elif len(feature_list) != feature_length:
                 warnings.warn(f"Inconsistent feature length for '{image_file}'. Expected {feature_length}, got {len(feature_list)}. Skipping file.")
                 continue # Skip this file

            # Store filename and flattened features
            data_rows.append({'filename': image_file, 'features': feature_list}) 
        else:
            # Log failure (specific error printed in extract_features_from_png)
            print(f"Feature extraction failed for image: '{image_file}'")

    # 6. Save Features to CSV
    if not data_rows:
        print("\nNo features were successfully extracted from any images.")
        return
        
    if feature_length is None:
        print("\nCould not determine feature length (no features extracted successfully). Cannot create CSV.")
        return

    print("\nProcessing extracted features for saving...")
    try:
        # Create DataFrame: ['filename', 'features' (list)]
        df = pd.DataFrame(data_rows)

        # Efficiently expand the 'features' list into separate columns
        feature_columns = [f'feature_{i}' for i in range(feature_length)]
        features_df = pd.DataFrame(df['features'].tolist(), index=df.index, columns=feature_columns)

        # Combine filename column with the new feature columns
        final_df = pd.concat([df['filename'], features_df], axis=1)

        # Define save path and ensure directory exists
        output_csv_path = os.path.join(OUTPUT_SUBDIR, OUTPUT_CSV_FILENAME)
        os.makedirs(OUTPUT_SUBDIR, exist_ok=True)

        # Save DataFrame to CSV
        final_df.to_csv(output_csv_path, index=False) 

        print(f"\nSuccessfully extracted features for {len(final_df)} images.")
        print(f"Output saved to: {output_csv_path}")
        print(f"CSV dimensions: {len(final_df)} rows, {len(final_df.columns)} columns")

    except Exception as e:
        print(f"\nError occurred while creating or saving the CSV file: {e}")

    print("\nFinished processing.")


# --- Execution Guard ---
if __name__ == '__main__':
    # Optional: Suppress specific warnings if needed
    # warnings.simplefilter(action='ignore', category=FutureWarning)
    main()

# Generate training, validation, and test splits for the NIH Chest X-ray dataset.
#
# This script creates the dataset splits according to the paper:
# - Random split: 70% train, 10% val, 20% test
# - Training set excludes all images with ANY unseen class label
# - Validation and test sets include all images (can have seen or unseen classes)
#
# The output files match the format of dataset_splits/train.txt, val.txt, and test.txt:
# Each line: images_XXX/filename.png 0 1 0 0 0 0 0 0 0 0 0 0 0 0
# Where the 14 binary values represent the one-hot encoding for the 14 classes.

import os
import numpy as np
import pandas as pd
import glob
from collections import defaultdict
import random


# Define the 14 disease classes in the same order as in dataset.py
CLASSES = [
    'Atelectasis',
    'Cardiomegaly',
    'Effusion',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pneumonia',
    'Pneumothorax',
    'Consolidation',
    'Edema',
    'Emphysema',
    'Fibrosis',
    'Pleural_Thickening',
    'Hernia'
]

# Seen and unseen classes as defined in the paper
SEEN_CLASSES = [
    'Atelectasis',
    'Effusion',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pneumothorax',
    'Consolidation',
    'Cardiomegaly',
    'Pleural_Thickening',
    'Hernia'
]

UNSEEN_CLASSES = [
    'Edema',
    'Pneumonia',
    'Emphysema',
    'Fibrosis'
]

# Create mapping from class name to index
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}


# Find all PNG image files in the dataset directory structure.
def find_image_paths(data_root):
    print(f"Searching for images in {data_root}...")

    # Find all PNG files in subdirectories
    pattern = os.path.join(data_root, '**', 'images', '*.png')
    image_paths = glob.glob(pattern, recursive=True)

    if not image_paths:
        # Try alternative pattern if images are directly in data_root
        pattern = os.path.join(data_root, '**', '*.png')
        image_paths = glob.glob(pattern, recursive=True)

    print(f"Found {len(image_paths)} image files")

    # Create mapping from filename to relative path
    filename_to_path = {}
    for full_path in image_paths:
        filename = os.path.basename(full_path)

        # Extract relative path from data_root
        rel_path = os.path.relpath(full_path, data_root)

        # Normalize path separators
        rel_path = rel_path.replace('\\', '/')

        filename_to_path[filename] = rel_path

    return filename_to_path


# Load image labels from the CSV file.
# Returns a dictionary mapping image filename to list of class names
def load_labels_from_csv(csv_path):
    print(f"Loading labels from {csv_path}...")
    df = pd.read_csv(csv_path)

    # First column is image filename, second column is labels (pipe-separated)
    image_to_labels = {}
    for _, row in df.iterrows():
        image_name = row.iloc[0]  # Image Index column
        labels_str = row.iloc[1]  # Finding Labels column

        # Split pipe-separated labels
        labels = labels_str.split('|')
        image_to_labels[image_name] = labels

    print(f"Loaded labels for {len(image_to_labels)} images")
    return image_to_labels


# Check if any of the labels are unseen classes.
def has_unseen_class(labels):
    return any(label in UNSEEN_CLASSES for label in labels)

# Convert a list of class names to a binary vector of length 14.
def labels_to_binary_vector(labels):
    binary = [0] * len(CLASSES)
    for label in labels:
        if label in CLASS_TO_IDX:
            binary[CLASS_TO_IDX[label]] = 1
    return binary


# Write a split file in the required format.
def write_split_file(output_path, image_data):
    with open(output_path, 'w') as f:
        for rel_path, binary_labels in image_data:
            # Format: path/filename.png 0 1 0 0 0 0 0 0 0 0 0 0 0 0
            binary_str = ' '.join(map(str, binary_labels))
            f.write(f"{rel_path} {binary_str}\n")

    print(f"Wrote {len(image_data)} images to {output_path}")


# Generate train/val/test splits according to the paper specifications.
# csv_path: Path to Data_Entry CSV file
# data_root: Root directory containing images
# output_dir: Directory to save split files
def generate_splits(csv_path, data_root, output_dir='dataset_splits'):
    seed = 1002  # Same seed as used in train.py for reproducibility
    train_ratio = 0.7
    val_ratio = 0.1
    test_ratio = 0.2

    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    print("Generating Dataset Splits for NIH Chest X-ray Dataset\n\n")

    # Load image paths and labels
    filename_to_path = find_image_paths(data_root)
    image_to_labels = load_labels_from_csv(csv_path)

    # Match images with their labels and paths
    valid_images = []
    for filename, labels in image_to_labels.items():
        # Skip images with only "No Finding"
        if len(labels) == 1 and labels[0] == 'No Finding':
            continue

        # Skip if image file not found
        if filename not in filename_to_path:
            continue

        rel_path = filename_to_path[filename]
        valid_images.append((filename, rel_path, labels))

    print(f"\nTotal valid images (excluding 'No Finding' only): {len(valid_images)}")

    # Separate images into those with and without unseen classes
    images_without_unseen = [] # Can be in training set
    images_with_unseen = [] # Cannot be in training set

    for filename, rel_path, labels in valid_images:
        if has_unseen_class(labels):
            images_with_unseen.append((filename, rel_path, labels))
        else:
            images_without_unseen.append((filename, rel_path, labels))

    print(f"Images without unseen classes (eligible for training): {len(images_without_unseen)}")
    print(f"Images with unseen classes (excluded from training): {len(images_with_unseen)}")

    # Shuffle both groups
    random.shuffle(images_without_unseen)
    random.shuffle(images_with_unseen)

    # Calculate target sizes based on total valid images
    total_valid = len(valid_images)
    target_train = int(total_valid * train_ratio)
    target_val = int(total_valid * val_ratio)
    target_test = total_valid - target_train - target_val  # Remaining goes to test

    print(f"\nTarget split sizes (based on {total_valid} total images):")
    print(f"  Train: {target_train} ({train_ratio*100:.1f}%)")
    print(f"  Val: {target_val} ({val_ratio*100:.1f}%)")
    print(f"  Test: {target_test} ({test_ratio*100:.1f}%)")

    # Training set: only images without unseen classes
    # Take up to target_train images from images_without_unseen
    n_train = min(target_train, len(images_without_unseen))
    train_images = images_without_unseen[:n_train]
    remaining_seen = images_without_unseen[n_train:]

    # Remaining images (both with and without unseen classes) go to val/test
    # Split them to maintain approximately 10/20 ratio
    remaining_total = len(remaining_seen) + len(images_with_unseen)

    # Calculate how many images we need for val and test
    # Try to maintain the val/test ratio (1:2)
    val_plus_test_ratio = val_ratio + test_ratio
    target_val_remaining = int(remaining_total * val_ratio / val_plus_test_ratio)
    target_test_remaining = remaining_total - target_val_remaining

    # Split remaining seen images between val and test
    n_val_seen = min(target_val_remaining, len(remaining_seen))
    val_images_from_seen = remaining_seen[:n_val_seen]
    test_images_from_seen = remaining_seen[n_val_seen:]

    # Calculate how many more images we need for val and test
    val_needed = max(0, target_val_remaining - len(val_images_from_seen))
    test_needed = max(0, target_test_remaining - len(test_images_from_seen))

    # Add images with unseen classes to val and test
    # Split them to fill remaining slots
    n_val_unseen = min(val_needed, len(images_with_unseen))
    val_images_from_unseen = images_with_unseen[:n_val_unseen]
    remaining_unseen = images_with_unseen[n_val_unseen:]

    # Add remaining unseen images to test
    test_images_from_unseen = remaining_unseen

    print(f"\nValidation set:")
    print(f"  seen classes: {len(val_images_from_seen)}")
    print(f"  unseen classes: {len(val_images_from_unseen)}")
    print(f"  total: {len(val_images_from_seen) + len(val_images_from_unseen)}")

    print(f"\nTest set:")
    print(f"  seen classes: {len(test_images_from_seen)}")
    print(f"  unseen classes: {len(test_images_from_unseen)}")
    print(f"  total: {len(test_images_from_seen) + len(test_images_from_unseen)}")

    # Combine val and test sets
    val_images = val_images_from_seen + val_images_from_unseen
    test_images = test_images_from_seen + test_images_from_unseen

    # Shuffle val and test sets
    random.shuffle(val_images)
    random.shuffle(test_images)

    print(f"\nFinal split sizes:")
    print(f"  Train: {len(train_images)}")
    print(f"  Val: {len(val_images)}")
    print(f"  Test: {len(test_images)}")
    print(f"  Total: {len(train_images) + len(val_images) + len(test_images)}")

    # Convert to format needed for output files
    train_data = [(rel_path, labels_to_binary_vector(labels)) 
                  for _, rel_path, labels in train_images]
    val_data = [(rel_path, labels_to_binary_vector(labels)) 
                for _, rel_path, labels in val_images]
    test_data = [(rel_path, labels_to_binary_vector(labels)) 
                 for _, rel_path, labels in test_images]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Write split files
    print(f"\nWriting split files to {output_dir}...")
    write_split_file(os.path.join(output_dir, 'train.txt'), train_data)
    write_split_file(os.path.join(output_dir, 'val.txt'), val_data)
    write_split_file(os.path.join(output_dir, 'test.txt'), test_data)

    print(f"\nFinal counts:")
    print(f"  Train: {len(train_data)}")
    print(f"  Val: {len(val_data)}")
    print(f"  Test: {len(test_data)}")

    return train_data, val_data, test_data


if __name__ == '__main__':
    # Configuration
    csv_path = 'CXR8/Data_Entry_2017_v2020.csv'
    data_root = 'CXR8'  # Root directory containing images
    output_dir = 'dataset_splits'

    # Check if CSV file exists
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        print("Please ensure the CSV file exists in the CXR8 directory.")
        return

    # Check if data root exists
    if not os.path.exists(data_root):
        print(f"Error: Data root directory not found at {data_root}")
        print("Please ensure the CXR8 directory exists.")
        return

    # Generate splits
    generate_splits(csv_path, data_root, output_dir)

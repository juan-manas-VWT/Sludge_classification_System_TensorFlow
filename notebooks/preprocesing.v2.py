#2 preprocessing without considering resize and related operations.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
import glob
from tqdm import tqdm  # Changed from tqdm.notebook to standard tqdm
import shutil
from collections import Counter

# Initial configuration
np.random.seed(42)
tf.random.set_seed(42)

# Define paths
RAW_DATA_DIR = 'data/raw/'
TRAIN_DIR = 'data/train/'
VALIDATION_DIR = 'data/validation/'
TEST_DIR = 'data/test/'


# Create directories
for directory in [TRAIN_DIR, VALIDATION_DIR, TEST_DIR]:  # Add TEST_DIR
    os.makedirs(directory, exist_ok=True)
    os.makedirs(os.path.join(directory, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(directory, 'anomaly'), exist_ok=True)

# ---------------------------------------------
# 1. Exploratory data analysis
# ---------------------------------------------

def analyze_dataset(data_dir):
    """
    Analyzes the dataset and returns basic statistics.
    
    Args:
        data_dir: Directory containing the images
    
    Returns:
        A dictionary with the statistics
    """
    stats = {}
    
    for category in ['normal', 'anomaly']:
        category_dir = os.path.join(data_dir, category)
        if not os.path.exists(category_dir):
            print(f"The directory {category_dir} does not exist.")
            continue
        
        # Count images
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_paths = []
        
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(category_dir, ext)))
        
        num_images = len(image_paths)
        stats[f'{category}_count'] = num_images
        
        # Analyze image sizes and properties
        if num_images > 0:
            # Sample up to 20 images to calculate sizes
            sample_size = min(20, num_images)
            sample_paths = np.random.choice(image_paths, sample_size, replace=False)
            
            image_sizes = []
            image_channels = []
            image_types = []
            
            for path in sample_paths:
                try:
                    img = cv2.imread(path)
                    if img is not None:
                        height, width = img.shape[:2]
                        channels = img.shape[2] if len(img.shape) > 2 else 1
                        image_sizes.append((height, width))
                        image_channels.append(channels)
                        image_types.append(img.dtype)
                except Exception as e:
                    print(f"Error reading {path}: {e}")
            
            if image_sizes:
                stats[f'{category}_sizes'] = image_sizes
                stats[f'{category}_channels'] = image_channels
                stats[f'{category}_types'] = image_types
                
                # Calculate statistics
                heights, widths = zip(*image_sizes)
                stats[f'{category}_height_mean'] = np.mean(heights)
                stats[f'{category}_width_mean'] = np.mean(widths)
                stats[f'{category}_height_min'] = min(heights)
                stats[f'{category}_height_max'] = max(heights)
                stats[f'{category}_width_min'] = min(widths)
                stats[f'{category}_width_max'] = max(widths)
                
                # Get the most common sizes
                stats[f'{category}_common_sizes'] = Counter(image_sizes).most_common(3)
    
    return stats

print("Analyzing original dataset...")
dataset_stats = analyze_dataset(RAW_DATA_DIR)

# Display statistics
total_images = dataset_stats.get('normal_count', 0) + dataset_stats.get('anomaly_count', 0)
print(f"\n--- Dataset Statistics ---")
print(f"Total images: {total_images}")
print(f"- Normal images: {dataset_stats.get('normal_count', 0)} ({dataset_stats.get('normal_count', 0)/total_images*100:.1f}%)")
print(f"- Anomaly images: {dataset_stats.get('anomaly_count', 0)} ({dataset_stats.get('anomaly_count', 0)/total_images*100:.1f}%)")

# Check for class imbalance
if min(dataset_stats.get('normal_count', 0), dataset_stats.get('anomaly_count', 0)) / max(dataset_stats.get('normal_count', 0), dataset_stats.get('anomaly_count', 0)) < 0.5:
    print("\n⚠️ ALERT: The dataset is imbalanced. Consider using class_weights during training.")

# Display size statistics
print("\n--- Size Statistics ---")
for category in ['normal', 'anomaly']:
    if f'{category}_common_sizes' in dataset_stats:
        print(f"\n{category.capitalize()}:")
        print(f"- Average dimensions: {dataset_stats[f'{category}_height_mean']:.0f}x{dataset_stats[f'{category}_width_mean']:.0f} pixels")
        print(f"- Height range: {dataset_stats[f'{category}_height_min']} - {dataset_stats[f'{category}_height_max']} pixels")
        print(f"- Width range: {dataset_stats[f'{category}_width_min']} - {dataset_stats[f'{category}_width_max']} pixels")
        print(f"- Most common sizes: {dataset_stats[f'{category}_common_sizes']}")

# ---------------------------------------------
# 2. Image examples visualization
# ---------------------------------------------

def load_and_show_examples(data_dir, num_examples=3):
    """Loads and displays examples of normal and anomaly images."""
    
    fig = plt.figure(figsize=(15, 8))
    
    for i, category in enumerate(['normal', 'anomaly']):
        category_dir = os.path.join(data_dir, category)
        if not os.path.exists(category_dir):
            continue
            
        # Search for all common extensions
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_paths.extend(glob.glob(os.path.join(category_dir, ext)))
        
        # Select random images
        if len(image_paths) > num_examples:
            image_paths = np.random.choice(image_paths, num_examples, replace=False)
        
        for j, path in enumerate(image_paths):
            try:
                img = cv2.imread(path)
                if img is not None:
                    # Convert from BGR to RGB for matplotlib
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Display image
                    ax = fig.add_subplot(2, num_examples, i * num_examples + j + 1)
                    ax.imshow(img_rgb)
                    ax.set_title(f"{category.capitalize()} - {img.shape}")
                    ax.axis('off')
            except Exception as e:
                print(f"Error loading {path}: {e}")
    
    plt.tight_layout()
    plt.show()

# Display image examples
print("\nLoading example images from the original dataset...")
load_and_show_examples(RAW_DATA_DIR, num_examples=4)

# ---------------------------------------------
# 3. Split into training, validation, and testing sets
# ---------------------------------------------

def split_dataset(raw_dir, train_dir, val_dir, test_dir, train_ratio=0.7, val_ratio=0.15):
    """
    Splits the dataset into training, validation, and test sets.
    
    Args:
        raw_dir: Directory with original images
        train_dir: Directory for the training set
        val_dir: Directory for the validation set
        test_dir: Directory for the test set
        train_ratio: Proportion of the training set (default 70%)
        val_ratio: Proportion of the validation set (default 15%)
        (implicitly, test_ratio = 1 - train_ratio - val_ratio)
    """
    split_stats = {'train': {}, 'validation': {}, 'test': {}}
    
    # Process each category (normal and anomaly)
    for category in ['normal', 'anomaly']:
        # Get the paths of the category images
        image_paths = glob.glob(os.path.join(raw_dir, category, '*.jpg')) + \
                     glob.glob(os.path.join(raw_dir, category, '*.png')) + \
                     glob.glob(os.path.join(raw_dir, category, '*.jpeg')) + \
                     glob.glob(os.path.join(raw_dir, category, '*.JPG')) + \
                     glob.glob(os.path.join(raw_dir, category, '*.PNG')) + \
                     glob.glob(os.path.join(raw_dir, category, '*.JPEG'))
        
        if not image_paths:
            print(f"No images found for category {category}")
            continue
        
        # Shuffle the images to ensure random distribution
        np.random.shuffle(image_paths)
        
        # Calculate amount for each set
        total_images = len(image_paths)
        train_size = int(total_images * train_ratio)
        val_size = int(total_images * val_ratio)
        
        # Split into three sets
        train_paths = image_paths[:train_size]
        val_paths = image_paths[train_size:train_size + val_size]
        test_paths = image_paths[train_size + val_size:]
        
        split_stats['train'][category] = len(train_paths)
        split_stats['validation'][category] = len(val_paths)
        split_stats['test'][category] = len(test_paths)
        
        # Copy images to the corresponding directories
        print(f"Copying {category} images...")
        
        # Training
        for path in tqdm(train_paths):
            dest_path = os.path.join(train_dir, category, os.path.basename(path))
            shutil.copy(path, dest_path)
        
        # Validation
        for path in tqdm(val_paths):
            dest_path = os.path.join(val_dir, category, os.path.basename(path))
            shutil.copy(path, dest_path)
            
        # Test (new)
        for path in tqdm(test_paths):
            dest_path = os.path.join(test_dir, category, os.path.basename(path))
            shutil.copy(path, dest_path)
    
    return split_stats

## DATASET DIVISION

# Split the dataset into three parts
print("\nSplitting the dataset into training, validation, and test sets...")
split_stats = split_dataset(RAW_DATA_DIR, TRAIN_DIR, VALIDATION_DIR, TEST_DIR, 
                          train_ratio=0.7, val_ratio=0.15)

# Display the results of the split
print("\n--- Split Results ---")
print("Training set:")
print(f"- Normal: {split_stats['train'].get('normal', 0)} images")
print(f"- Anomaly: {split_stats['train'].get('anomaly', 0)} images")
print(f"- Total: {sum(split_stats['train'].values())} images")

print("\nValidation set:")
print(f"- Normal: {split_stats['validation'].get('normal', 0)} images")
print(f"- Anomaly: {split_stats['validation'].get('anomaly', 0)} images")
print(f"- Total: {sum(split_stats['validation'].values())} images")

print("\nTest set:")  # New
print(f"- Normal: {split_stats['test'].get('normal', 0)} images")
print(f"- Anomaly: {split_stats['test'].get('anomaly', 0)} images")
print(f"- Total: {sum(split_stats['test'].values())} images")

# ---------------------------------------------
# 4. Save configuration for fine-tuning
# ---------------------------------------------

# Calculate parameters for training
BATCH_SIZE = 32
IMG_SIZE = (224, 224)  # Common size for pretrained models

# Calculate steps per epoch
train_total = sum(split_stats['train'].values())
val_total = sum(split_stats['validation'].values())

steps_per_epoch = train_total // BATCH_SIZE
validation_steps = val_total // BATCH_SIZE

# Create a dictionary with the fine-tuning configuration
# Where the configuration dictionary is created
config = {
    'image_size': IMG_SIZE,
    'batch_size': BATCH_SIZE,
    'train_samples': sum(split_stats['train'].values()),
    'validation_samples': sum(split_stats['validation'].values()),
    'test_samples': sum(split_stats['test'].values()),  # New
    'steps_per_epoch': steps_per_epoch,
    'validation_steps': validation_steps,
    'class_indices': {'anomaly': 0, 'normal': 1},
    'data_augmentation': {
        'rotation_range': 20,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'shear_range': 0.2,
        'zoom_range': 0.2,
        'horizontal_flip': True,
        'fill_mode': 'nearest'
    }
}
# Save the configuration as JSON
import json
os.makedirs('data/', exist_ok=True)
config_path = 'data/data_config.json'

with open(config_path, 'w') as f:
    json.dump(config, f, indent=4)

print(f"\nConfiguration saved in {config_path}")

# Display examples from training and validation sets
print("\nShowing examples from the training set:")
load_and_show_examples(TRAIN_DIR, num_examples=3)

print("\nShowing examples from the validation set:")
load_and_show_examples(VALIDATION_DIR, num_examples=3)

print("\nPreprocessing completed successfully!")
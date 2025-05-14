import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

# Define paths
TRAIN_DIR = 'data/train/'
VALIDATION_DIR = 'data/validation/'

# New img size after cropping to follow the rectangular shape
IMG_SIZE = (186, 400)  # height x width

def preprocess_image(img):
    """
    Crops a specific region of the image and resizes it to the desired size.
    If the image is too small, the crop is adapted proportionally.
    """
    # Get image dimensions
    img_height, img_width = img.shape[:2]
    
    # Original coordinates for cropping (for full-size images)
    target_start_x = 943
    target_start_y = 861
    target_width = 3569
    target_height = 1651
    
    # Calculate the proportion of the current image with respect to a reference size
    # (assuming that the original coordinates are for 4512x2512 images)
    reference_width = 4512
    reference_height = 2512
    
    width_ratio = img_width / reference_width
    height_ratio = img_height / reference_height
    
    # Adjust coordinates according to the proportion
    start_x = int(target_start_x * width_ratio)
    start_y = int(target_start_y * height_ratio)
    width = int(target_width * width_ratio)
    height = int(target_height * height_ratio)
    
    # Ensure coordinates are within image boundaries
    start_x = max(0, min(start_x, img_width - 1))
    start_y = max(0, min(start_y, img_height - 1))
    width = min(width, img_width - start_x)
    height = min(height, img_height - start_y)
    
    # If the crop area is too small, use the full image
    if width < 50 or height < 50:
        print(f"Image too small to crop: {img_width}x{img_height}")
        return cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0]))
    
    # Perform the crop
    end_x = start_x + width
    end_y = start_y + height
    cropped_img = img[start_y:end_y, start_x:end_x]
    
    # Resize the cropped image
    return cv2.resize(cropped_img, (IMG_SIZE[1], IMG_SIZE[0]))

def visualize_preprocessing(image_path, save_path=None):
    """
    Visualize how the preprocessing affects the input images using exact crop coordinates.
    
    Args:
        image_path: Path to a sample image
        save_path: Optional path to save the visualization
    """
    # Read the original image
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Could not read image at {image_path}")
        return None, None, None, None
        
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Use exact coordinates for cropping
    start_x = 943
    start_y = 861
    width = 3569
    height = 1651
    
    # Ensure coordinates are within image boundaries
    img_height, img_width = original_img.shape[:2]
    
    # Print original image dimensions for reference
    print(f"Original image dimensions: {img_width}x{img_height}")
    
    # Adjust if necessary to prevent out-of-bounds errors
    start_x = min(start_x, img_width - 1)
    start_y = min(start_y, img_height - 1)
    width = min(width, img_width - start_x)
    height = min(height, img_height - start_y)
    
    end_x = start_x + width
    end_y = start_y + height
    
    # Create a copy of the original image with the ROI highlighted
    highlighted_img = original_img.copy()
    cv2.rectangle(highlighted_img, (start_x, start_y), (end_x, end_y), (255, 0, 0), 5)
    
    # Crop the image
    cropped_img = original_img[start_y:end_y, start_x:end_x]
    
    # Resize the cropped image to model input size
    resized_img = cv2.resize(cropped_img, (IMG_SIZE[1], IMG_SIZE[0]))
    
    # Create the figure for visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Plot each step
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(highlighted_img)
    axes[1].set_title(f'Region of Interest\n({start_x},{start_y}, {width}x{height})')
    axes[1].axis('off')
    
    axes[2].imshow(cropped_img)
    axes[2].set_title('Cropped Image')
    axes[2].axis('off')
    
    axes[3].imshow(resized_img)
    axes[3].set_title(f'Resized to {IMG_SIZE[0]}x{IMG_SIZE[1]}')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return original_img, highlighted_img, cropped_img, resized_img

def test_with_sample_image():
    """
    Test the preprocessing with a sample image to verify the crop coordinates.
    """
    # Find a sample image
    sample_paths = []
    for root, _, files in os.walk(TRAIN_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                sample_paths.append(os.path.join(root, file))
                break
        if sample_paths:
            break
    
    if not sample_paths:
        print("No sample images found for testing.")
        return
    
    sample_path = sample_paths[0]
    print(f"Testing with sample image: {sample_path}")
    
    # Visualize the preprocessing
    visualize_preprocessing(sample_path, 'crop_test.png')
    
    print("Test complete.")

def analyze_batch_images(batch_x, batch_y, num_samples=5):
    """
    Detailed analysis of processed images to diagnose
    visualization problems.
    """
    print("\n--- DETAILED IMAGE ANALYSIS ---")
    
    # Global batch statistics
    print(f"Batch shape: {batch_x.shape}")
    print(f"Batch dtype: {batch_x.dtype}")
    print(f"Global range: [{batch_x.min():.4f}, {batch_x.max():.4f}]")
    
    # Create figure for visualization
    plt.figure(figsize=(18, 12))
    
    for i in range(min(num_samples, batch_x.shape[0])):
        img = batch_x[i]
        label = batch_y[i]
        
        # Image statistics
        print(f"\nImage {i+1} (Class {int(label)}):")
        print(f"  Range: [{img.min():.4f}, {img.max():.4f}]")
        print(f"  Mean: {img.mean():.4f}")
        print(f"  Standard deviation: {img.std():.4f}")
        
        # Histogram of values
        values_flat = img.reshape(-1)
        
        # Original visualization
        plt.subplot(num_samples, 4, i*4 + 1)
        plt.imshow(img)
        plt.title(f"Original (Class {int(label)})")
        plt.axis('off')
        
        # Histogram
        plt.subplot(num_samples, 4, i*4 + 2)
        plt.hist(values_flat, bins=50, color='blue', alpha=0.7)
        plt.title(f"Histogram\nRange: [{img.min():.2f}, {img.max():.2f}]")
        plt.grid(True, alpha=0.3)
        
        # Visualization with adjusted contrast
        plt.subplot(num_samples, 4, i*4 + 3)
        plt.imshow(img, vmin=0.0, vmax=0.5)  # Adjust maximum for better visualization
        plt.title("Adjusted contrast\nvmax=0.5")
        plt.axis('off')
        
        # CLAHE version for comparison
        img_uint8 = (img * 255).astype(np.uint8)
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        enhanced_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        plt.subplot(num_samples, 4, i*4 + 4)
        plt.imshow(enhanced_img)
        plt.title("CLAHE applied")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def get_data_generators(preprocessing_func=preprocess_image, batch_size=16):
    """
    Creates and configures data generators for training and validation.
    
    Args:
        preprocessing_func: Preprocessing function to apply
        batch_size: Batch size for generators
        
    Returns:
        Tuple with (train_generator, validation_generator, class_indices, class_weight)
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.utils.class_weight import compute_class_weight
    
    # Data generator for training with data augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_func,  # Custom preprocessing applied first
        rescale=1./255,    # Normalize here
        rotation_range=5,           # Reduced rotation for sludge images
        # width_shift_range=0.1,      # Horizontal shift
        # height_shift_range=0.1,     # Vertical shift
        # zoom_range=0.15,            # Zoom range
        horizontal_flip=True,       # Horizontal flip
        # fill_mode='nearest'         # Fill mode for augmentation
    )

    # Data generator for validation (only preprocessing, no augmentation)
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_func, # Only apply the custom preprocessing
        rescale=1./255,    # Normalize here
    )

    # Prepare the data generators with the specified target size
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,  # This will be applied after preprocessing
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )

    validation_generator = val_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=IMG_SIZE,  # This will be applied after preprocessing
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    # Get class indices from the generator
    class_indices = train_generator.class_indices
    print(f"Class indices: {class_indices}")
    
    # Calculate class weights to handle imbalance
    class_counts = [0, 0]
    for i in range(len(train_generator.classes)):
        class_counts[train_generator.classes[i]] += 1

    print(f"Class distribution in training set: {class_counts}")

    # Calculate weights inversely proportional to class frequency
    if class_counts[0] != class_counts[1]:
        total = sum(class_counts)
        class_weight = {
            0: total / (2 * class_counts[0]),
            1: total / (2 * class_counts[1])
        }
        print(f"Calculated class weights: {class_weight}")
    else:
        class_weight = None
        print("Classes are balanced. No class weights needed.")
        
    # Calculate parameters based on directory size
    train_samples = sum([len(files) for _, _, files in os.walk(TRAIN_DIR)])
    validation_samples = sum([len(files) for _, _, files in os.walk(VALIDATION_DIR)])
    steps_per_epoch = train_samples // batch_size
    validation_steps = validation_samples // batch_size
    
    print(f"Train samples: {train_samples}, validation samples: {validation_samples}")
    print(f"Steps per epoch: {steps_per_epoch}, validation steps: {validation_steps}")
    
    return (train_generator, validation_generator, class_indices, 
            class_weight, steps_per_epoch, validation_steps)

# Main function to test the preprocessing module
if __name__ == "__main__":
    # Test the preprocessing with a sample image
    test_with_sample_image()
    
    # Get data generators to test them
    (train_generator, validation_generator, 
     class_indices, class_weight, 
     steps_per_epoch, validation_steps) = get_data_generators()
    
    # Verify preprocessing and normalization
    print("Verifying preprocessing:")
    batch_x, batch_y = next(train_generator)
    print(f"Batch shape: {batch_x.shape}")  # Should be (BATCH_SIZE, HEIGHT, WIDTH, 3)
    print(f"Value range: [{batch_x.min()}, {batch_x.max()}]")  # Should be close to [0, 1]

    # Display a few processed training images to verify preprocessing
    plt.figure(figsize=(15, 5))
    for i in range(min(5, batch_x.shape[0])):
        plt.subplot(1, 5, i+1)
        plt.imshow(batch_x[i])
        plt.title(f"Class: {int(batch_y[i])}")
        plt.axis('off')
    plt.suptitle("Processed Training Images")
    plt.tight_layout()
    plt.show()
    
    # Run detailed analysis on batch images
    analyze_batch_images(batch_x, batch_y)
import os
import sys
import zipfile
import shutil
import io # Needed for nested zip handling (kept for robustness)
import glob # Needed for flexible data flattening

# --- Configuration (Global Constants) ---
# New data directory configuration
DATA_DIR_BASE = 'biomedical_data'
# The directory where the flattened, Keras-ready dataset will be placed
DATA_DIR_RESIZED = os.path.join("..", "data", "biomedical_dataset_FLAT") 
# Assuming the raw, unzipped folder structure is in ..\data\biomedical_dataset
RAW_DATA_DIR = os.path.join("..", "data", "biomedical_dataset")

SEED = 42 # Used for reproducibility

# Image and Data Loading Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

# Class names found from your training run (12 classes)
MOCK_CATEGORIES = [
    '(BT) Body Tissue or Organ', 
    '(GE) Glass equipment-packaging 551', 
    '(ME) Metal equipment -packaging', 
    '(OW) Organic wastes', 
    '(PE) Plastic equipment-packaging', 
    '(PP) Paper equipment-packaging', 
    '(SN) Syringe needles', 
    'Gauze', 
    'Gloves', 
    'Mask', 
    'Syringe', 
    'Tweezers'
]

# --- Dependency Handling (Mocking for portability) ---
try:
    # Attempt to import the actual packages
    import tensorflow as tf
    import numpy as np # Needed for image handling in predict.py
    from tensorflow.keras.utils import image_dataset_from_directory
    
    # Check for numpy and tensorflow to ensure environment is complete
    if not hasattr(tf, 'version') or not hasattr(np, 'array'):
        raise ImportError("TensorFlow or NumPy not fully initialized.")

    print("TensorFlow data utilities available for use.")
    IS_REAL_TF = True

except (ImportError, AttributeError):
    # Define mocks if TensorFlow is not installed
    class MockTensorFlow:
        # Mock for tf.version
        version = 'MOCK_2.X' 
        
        # Mock for tf.random.set_seed
        def random_set_seed(self, seed):
            pass

        # Mock for the data layer utility
        def keras_utils_image_dataset_from_directory(self, directory, labels, validation_split, subset, seed, image_size, batch_size, label_mode):
            """Mock dataset generator that prints messages and returns a mock object."""
            # Use the global MOCK_CATEGORIES for class count
            num_files = 6586 # Mock value based on your run
            num_classes = len(MOCK_CATEGORIES)
            
            print(f"Dynamically detected classes ({num_classes}): {MOCK_CATEGORIES}")
            print(f"Found {num_files} files belonging to {num_classes} classes.")
            
            if subset == "training":
                num_subset = int(num_files * (1 - VALIDATION_SPLIT))
                print(f"Using {num_subset} files for training.")
            elif subset == "validation":
                num_subset = int(num_files * VALIDATION_SPLIT)
                print(f"Using {num_subset} files for validation.")

            class MockDataset:
                def __init__(self, names):
                    self.class_names = names
                    
                # Mock cache and prefetch methods for compatibility
                def cache(self): return self
                def prefetch(self, buffer_size): return self

            return MockDataset(MOCK_CATEGORIES)

    # Assign mocks to variables used in the script
    tf = MockTensorFlow()
    tf.random.set_seed = tf.random_set_seed
    tf.keras = MockTensorFlow()
    tf.keras.utils = MockTensorFlow()
    tf.keras.utils.image_dataset_from_directory = tf.keras_utils_image_dataset_from_directory
    np = MockTensorFlow()
    np.expand_dims = lambda x, axis: x # Mock numpy expand_dims

    print("TensorFlow data utilities not found. Using mock utilities.")
    IS_REAL_TF = False


# --- Data Preparation Functions ---

def flatten_dataset_structure(source_dir, target_dir):
    """
    Copies all image files from nested subdirectories into a flat Keras-compatible structure:
    target_dir/class_name/image_name.jpg
    This function detects classes based on the folders in the raw data directory.
    """
    print(f"\n--- Flattening Dataset Structure for Keras compatibility (Flexible Search) ---")
    
    # Clean up existing flat directory
    if os.path.exists(target_dir):
        print(f"Removing existing flat directory: {target_dir}")
        shutil.rmtree(target_dir, ignore_errors=True)
    
    os.makedirs(target_dir, exist_ok=True)
    
    total_files_moved = 0
    # Search for all subdirectories in the raw data directory (e.g., 'Gloves', 'Mask')
    raw_class_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    # 1. Update MOCK_CATEGORIES globally based on the detected directories
    global MOCK_CATEGORIES
    if raw_class_dirs:
        MOCK_CATEGORIES = raw_class_dirs
        
    for class_name in raw_class_dirs:
        raw_class_path = os.path.join(source_dir, class_name)
        
        # Create the corresponding directory in the target FLAT structure
        target_class_path = os.path.join(target_dir, class_name)
        os.makedirs(target_class_path, exist_ok=True)
        
        # Search for images recursively in the raw class directory
        for root, _, files in os.walk(raw_class_path):
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    source_file = os.path.join(root, filename)
                    # Use a unique name to avoid conflicts, including class name and original filename
                    target_file_name = f"{class_name}_{filename}"
                    target_file = os.path.join(target_class_path, target_file_name)
                    
                    try:
                        shutil.copy(source_file, target_file)
                        total_files_moved += 1
                    except Exception as e:
                        print(f"Error copying {source_file}: {e}")

        print(f"Processing category: {class_name}")

    print(f"--- Flattening Complete. Total files moved: {total_files_moved} ---")

def cleanup_extracted_data():
    """Removes the temporary extracted data directory."""
    pass # We are no longer using temporary extraction, skipping this.


def load_and_prepare_data():
    """
    Entry point for data loading. Handles data structure flattening 
    and returns TensorFlow datasets (or mocks).
    """
    print(f"Attempting to load data from {RAW_DATA_DIR}...")
    
    # Check if the raw data directory exists
    if not os.path.exists(RAW_DATA_DIR):
        print(f"Error: Raw data directory not found at {RAW_DATA_DIR}. Cannot proceed.")
        print("Please ensure your 'biomedical_dataset' folder is correctly placed in the 'data' directory.")
        return None, None
    
    print("\n--- Starting Data Preparation for Biomedical Dataset ---")
    
    # 1. Ensure Keras-compatible structure exists (Flatten the data)
    flatten_dataset_structure(RAW_DATA_DIR, DATA_DIR_RESIZED)
    
    # 2. Recheck classes using the globally updated list
    num_classes = len(MOCK_CATEGORIES)
    if num_classes == 0:
        print("Error: No classes detected after flattening.")
        return None, None

    # --- Data Loading Configuration uses global constants now ---
    train_ds, val_ds = None, None
    
    # Run only if the real TF is installed, otherwise the mock runs in the except block below.
    if IS_REAL_TF:
        # 1. Load Training Dataset
        train_ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR_RESIZED,
            labels='inferred',
            validation_split=VALIDATION_SPLIT,
            subset="training",
            seed=SEED,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='categorical'
        )
        
        # 2. Load Validation Dataset
        val_ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR_RESIZED,
            labels='inferred',
            validation_split=VALIDATION_SPLIT,
            subset="validation", # <-- FIXED: Removed erroneous backslash here
            seed=SEED,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='categorical'
        )

        # Print inferred class names (useful for debugging)
        if hasattr(train_ds, 'class_names'):
            class_names = [c for c in train_ds.class_names]
            print(f"\nInferred Class Names by Keras: {class_names}")

        print("Data loading complete.")

        # Prefetch data to improve training performance
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    else:
        # For mock environment, we still call the mock utility to print stats and return mock objects
        train_ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR_RESIZED, labels='inferred', validation_split=VALIDATION_SPLIT, 
            subset="training", seed=SEED, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR_RESIZED, labels='inferred', validation_split=VALIDATION_SPLIT, 
            subset="validation", seed=SEED, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
        )
        print("Data loading complete.")

    # Explicitly return the datasets/mock datasets
    return train_ds, val_ds

# Utility function for predict.py
def find_random_test_image(data_dir):
    """Finds a random image file in the flattened dataset for testing."""
    all_images = []
    # os.walk will find all files recursively within the target directory
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                all_images.append(os.path.join(root, file))

    if not all_images:
        return None
        
    import random
    random_image = random.choice(all_images)
    print(f"Test image path found: {random_image}")
    return random_image

if __name__ == "__main__":
    train_ds, val_ds = load_and_prepare_data()
    
    if train_ds is not None and val_ds is not None:
        # Try to find a test image for confirmation
        find_random_test_image(DATA_DIR_RESIZED)
    else:
        print("Data preparation failed.")
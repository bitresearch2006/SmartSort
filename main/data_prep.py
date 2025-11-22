import os
import sys
import zipfile
import shutil
import io # Needed for nested zip handling (kept for robustness)
import glob # Needed for flexible data flattening
import json
from datetime import datetime

# Use modular configuration
from config_loader import (
    get_data_config, get_model_config, get_image_size, 
    update_categories, load_categories
)

# Load configuration from JSON files
DATA_CONFIG = get_data_config()
MODEL_CONFIG = get_model_config()

# Extract configuration values
DATA_DIR_RESIZED = DATA_CONFIG.get('processed_data_dir', "../data/biomedical_dataset_FLAT")
RAW_DATA_DIR = DATA_CONFIG.get('raw_data_dir', "../data/biomedical_dataset")
IMAGE_SIZE = get_image_size()
BATCH_SIZE = MODEL_CONFIG.get('batch_size', 32)
VALIDATION_SPLIT = MODEL_CONFIG.get('validation_split', 0.2)
SEED = MODEL_CONFIG.get('seed', 42)
CATEGORIES_JSON_FILE = DATA_CONFIG.get('categories_file', 'categories.json')

# Categories will be dynamically detected and updated
MOCK_CATEGORIES = []

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


# --- Categories Management Functions ---

def count_images_per_category(source_dir, categories):
    """
    Count the number of images in each category directory.
    
    Args:
        source_dir (str): Path to the source data directory
        categories (list): List of category names
        
    Returns:
        dict: Dictionary with category names as keys and image counts as values
    """
    category_counts = {}
    total_images = 0
    
    for category in categories:
        category_path = os.path.join(source_dir, category)
        if os.path.exists(category_path):
            count = 0
            # Count image files recursively
            for root, _, files in os.walk(category_path):
                for filename in files:
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        count += 1
            category_counts[category] = count
            total_images += count
            print(f"   📊 {category}: {count} images")
        else:
            category_counts[category] = 0
            print(f"   ⚠️  {category}: Directory not found")
    
    print(f"   📈 Total images: {total_images}")
    return category_counts, total_images

def save_categories_to_json(categories, file_path=CATEGORIES_JSON_FILE):
    """
    Save the detected categories to a JSON file with metadata and image counts.
    
    Args:
        categories (list): List of category names
        file_path (str): Path to save the JSON file
    """
    print(f"📊 Counting images per category...")
    category_counts, total_images = count_images_per_category(RAW_DATA_DIR, categories)
    
    categories_data = {
        "categories": categories,
        "num_classes": len(categories),
        "total_images": total_images,
        "category_counts": category_counts,
        "last_updated": datetime.now().isoformat(),
        "source_directory": RAW_DATA_DIR,
        "description": "Biomedical waste classification categories detected during data preparation"
    }
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(categories_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Categories saved to {file_path}")
        print(f"   - Number of categories: {len(categories)}")
        print(f"   - Total images: {total_images}")
        print(f"   - Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        print(f"❌ Error saving categories to JSON: {e}")

def load_categories_from_json(file_path=CATEGORIES_JSON_FILE):
    """
    Load categories from JSON file.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        list: List of category names, or empty list if file doesn't exist
    """
    if not os.path.exists(file_path):
        print(f"📋 Categories JSON file not found: {file_path}")
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            categories_data = json.load(f)
        
        categories = categories_data.get('categories', [])
        print(f"✅ Categories loaded from {file_path}")
        print(f"   - Number of categories: {len(categories)}")
        print(f"   - Created: {categories_data.get('created_date', 'Unknown')}")
        print(f"   - Categories: {categories}")
        
        return categories
    except Exception as e:
        print(f"❌ Error loading categories from JSON: {e}")
        return []

def detect_categories_from_directory(source_dir):
    """
    Detect category names from directory structure.
    
    Args:
        source_dir (str): Path to the source data directory
        
    Returns:
        list: List of detected category names
    """
    if not os.path.exists(source_dir):
        print(f"❌ Source directory not found: {source_dir}")
        return []
    
    try:
        # Get all subdirectories as categories
        categories = [d for d in os.listdir(source_dir) 
                     if os.path.isdir(os.path.join(source_dir, d))]
        categories.sort()  # Sort for consistency
        
        print(f"🔍 Detected {len(categories)} categories from directory structure:")
        for i, category in enumerate(categories, 1):
            print(f"   {i:2d}. {category}")
        
        return categories
    except Exception as e:
        print(f"❌ Error detecting categories: {e}")
        return []

def update_categories():
    """
    Update the global MOCK_CATEGORIES by always scanning the current directory structure.
    This ensures the JSON file reflects the actual current dataset.
    """
    global MOCK_CATEGORIES
    
    print("\n🏷️  CATEGORY MANAGEMENT")
    print("=" * 50)
    
    # Always detect categories from the current directory structure
    print("🔍 Scanning directory structure for current categories...")
    categories = detect_categories_from_directory(RAW_DATA_DIR)
    
    if categories:
        # Always update/create the JSON file with current data
        print("� Updating categories JSON with current dataset structure...")
        save_categories_to_json(categories)
        
        # Update global categories
        MOCK_CATEGORIES = categories
        print(f"✅ Categories updated successfully: {len(categories)} categories found")
        return True
    else:
        print("❌ No categories could be detected from directory structure!")
        print(f"   Please check if the directory exists: {RAW_DATA_DIR}")
        MOCK_CATEGORIES = []
        return False

# --- Data Preparation Functions ---

def flatten_dataset_structure(source_dir, target_dir):
    """
    Copies all image files from nested subdirectories into a flat Keras-compatible structure:
    target_dir/class_name/image_name.jpg
    This function uses the global MOCK_CATEGORIES for class names.
    """
    print(f"\n📁 FLATTENING DATASET STRUCTURE")
    print("=" * 50)
    
    # Clean up existing flat directory
    if os.path.exists(target_dir):
        print(f"🗑️  Removing existing flat directory: {target_dir}")
        shutil.rmtree(target_dir, ignore_errors=True)
    
    os.makedirs(target_dir, exist_ok=True)
    
    total_files_moved = 0
    
    # Use the global MOCK_CATEGORIES (should be set by update_categories())
    if not MOCK_CATEGORIES:
        print("❌ No categories available for flattening!")
        return 0
        
    for class_name in MOCK_CATEGORIES:
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

        print(f"📂 Processing category: {class_name}")

    print(f"✅ Flattening Complete. Total files moved: {total_files_moved}")
    
    # Update categories JSON with current statistics
    if total_files_moved > 0:
        save_categories_to_json(MOCK_CATEGORIES)
    
    return total_files_moved

def cleanup_extracted_data():
    """Removes the temporary extracted data directory."""
    pass # We are no longer using temporary extraction, skipping this.


def load_and_prepare_data():
    """
    Entry point for data loading. Handles data structure flattening 
    and returns TensorFlow datasets (or mocks).
    """
    print(f"\n🚀 STARTING DATA PREPARATION")
    print("=" * 60)
    print(f"📂 Source directory: {RAW_DATA_DIR}")
    print(f"📁 Target directory: {DATA_DIR_RESIZED}")
    
    # Check if the raw data directory exists
    if not os.path.exists(RAW_DATA_DIR):
        print(f"❌ Raw data directory not found at {RAW_DATA_DIR}")
        print("Please ensure your 'biomedical_dataset' folder is correctly placed in the 'data' directory.")
        return None, None
    
    # 1. Update categories (load from JSON or detect from directories)
    if not update_categories():
        print("❌ Failed to update categories. Aborting data preparation.")
        return None, None
    
    # 2. Ensure Keras-compatible structure exists (Flatten the data)
    total_files = flatten_dataset_structure(RAW_DATA_DIR, DATA_DIR_RESIZED)
    
    if total_files == 0:
        print("❌ No files were processed during flattening.")
        return None, None
    
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
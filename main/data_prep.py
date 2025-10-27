import os
import sys
import zipfile
import requests
import shutil
import io

# --- Configuration (Global Constants) ---
GITHUB_ZIP_URL = "https://github.com/bitresearch2006/trashnet/archive/refs/heads/master.zip"

DATA_DIR_BASE = 'trashnet_data'
DATA_DIR_RESIZED = os.path.join(DATA_DIR_BASE, 'dataset-resized')
ZIP_FILENAME = os.path.join(DATA_DIR_BASE, 'trashnet-master.zip')
SEED = 42

# Image and Data Loading Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

# Class names of the dataset (6 classes)
# These are used to confirm we found the correct directory
MOCK_CATEGORIES = [
    "Cardboard",
    "Glass",
    "Metal",
    "Paper",
    "Plastic",
    "Trash"
]

# --- Dependency Handling (Mocking for portability) ---
try:
    # Attempt to import the actual packages
    import tensorflow as tf
    import numpy as np
    from tensorflow.keras.utils import image_dataset_from_directory
    
    print("TensorFlow data utilities available for use.")

except ImportError:
    print("TensorFlow or NumPy not found. Using mock data utilities.")
    
    # --- Mock objects for users without TensorFlow ---
    class MockTF:
        class MockKerasUtils:
            def image_dataset_from_directory(self, *args, **kwargs):
                print(f"Mock: Simulating data loading from {kwargs.get('directory', 'unknown')}")
                class MockDataset:
                    @property
                    def class_names(self):
                        return [c.capitalize() for c in MOCK_CATEGORIES] 
                return MockDataset(), MockDataset()

        class MockKeras:
            utils = MockKerasUtils()
            class MockLayer:
                def __init__(self, *args, **kwargs): pass
            
            def Input(self, *args, **kwargs): return self.MockLayer()

        def __init__(self):
            self.keras = self.MockKeras()
            self.random = self.MockRandom()
            self.version = "2.x"

        class MockRandom:
            def set_seed(self, seed):
                print(f"Mock: Setting random seed to {seed}")
        
    class MockNP:
        def __init__(self): pass
        def array(self, data): return data
        def random_sample(self, size): return [0.5] * size

    tf = MockTF()
    np = MockNP()


# --- Core Data Functions ---

def download_data():
    """Downloads the dataset zip file from GitHub."""
    os.makedirs(DATA_DIR_BASE, exist_ok=True)
    print(f"Downloading dataset from {GITHUB_ZIP_URL}...")
    
    try:
        response = requests.get(GITHUB_ZIP_URL, stream=True)
        response.raise_for_status()
        with open(ZIP_FILENAME, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Download complete. File saved to {ZIP_FILENAME}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error during download: {e}")
        return False

def cleanup_extracted_data():
    """Removes the initial zip file and the top-level extracted folder if it exists."""
    # Remove the zip file
    if os.path.exists(ZIP_FILENAME):
        os.remove(ZIP_FILENAME)

    # Remove the temporary extraction directory (this handles all nested temp files)
    temp_extract_dir = os.path.join(DATA_DIR_BASE, 'trashnet-master')
    if os.path.exists(temp_extract_dir):
        shutil.rmtree(temp_extract_dir)

def find_inner_zip(root_dir, target_name="dataset-resized.zip"):
    """Recursively searches for the inner dataset zip file."""
    print(f"Searching for inner ZIP: '{target_name}' inside {root_dir}...")
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if target_name in filenames:
            return os.path.join(dirpath, target_name)
    return None

def find_dataset_root(root_dir):
    """
    Recursively searches for a directory containing all the MOCK_CATEGORIES folders.
    Returns the path of the directory that is the true root of the class data.
    """
    expected_folders = set(c.lower() for c in MOCK_CATEGORIES)

    print(f"Searching for 6 class folders inside: {root_dir}")
    
    # os.walk traverses the directory tree
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Convert directory names found at this level to a set of lowercase names
        current_folders = set(d.lower() for d in dirnames)
        
        # Check if ALL expected class folders are present at this level
        if expected_folders.issubset(current_folders):
            print(f"Success! Found all 6 class folders inside: {dirpath}")
            return dirpath # This is the path we want to move

    return None

def extract_data():
    """Extracts the double-zipped file, finds the nested class data, and moves it to the final destination."""
    
    temp_extract_dir = os.path.join(DATA_DIR_BASE, 'trashnet-master')
    inner_extract_dir = os.path.join(temp_extract_dir, 'inner_data_temp')
    
    try:
        # 1. Extract Outer ZIP
        with zipfile.ZipFile(ZIP_FILENAME, 'r') as zip_ref:
            print(f"Extracting main ZIP to: {DATA_DIR_BASE}")
            zip_ref.extractall(DATA_DIR_BASE)

        # 2. Find Inner ZIP
        inner_zip_path = find_inner_zip(temp_extract_dir)
        if not inner_zip_path:
            print("Error: Could not find the required 'dataset-resized.zip' file within the extracted structure.")
            return False

        print(f"Found inner dataset ZIP: {inner_zip_path}")
        
        # 3. Extract Inner ZIP
        os.makedirs(inner_extract_dir, exist_ok=True)
        print(f"Extracting inner dataset to: {inner_extract_dir}")
        
        with zipfile.ZipFile(inner_zip_path, 'r') as inner_zip_ref:
            inner_zip_ref.extractall(inner_extract_dir)
            
        # 4. Find the actual root directory (the folder containing the six classes)
        # Search inside the inner extraction directory
        source_dir = find_dataset_root(inner_extract_dir) 
        
        if source_dir:
            # 5. Move contents
            # Ensure the final target directory exists
            os.makedirs(DATA_DIR_RESIZED, exist_ok=True)
            
            print(f"Moving class folders from {source_dir} to {DATA_DIR_RESIZED}...")
            
            # Move contents (the six class folders)
            for item in os.listdir(source_dir):
                s = os.path.join(source_dir, item)
                d = os.path.join(DATA_DIR_RESIZED, item)
                
                if os.path.isdir(s): # Only move the class folders
                    # We use shutil.move to move the directory itself
                    shutil.move(s, d)
            
            print("Dataset moved to correct structure.")
            return True
        else:
            print("Error: Could not find the 6 class folders after extracting the inner ZIP.")
            return False
            

    except zipfile.BadZipFile:
        print(f"Error: A zip file is corrupted or bad.")
        return False
    except Exception as e:
        print(f"Error during extraction: {e}")
        return False
    finally:
        # Clean up the zip file and all temporary extraction folders
        cleanup_extracted_data()


def load_and_prepare_data():
    """
    Checks for data, downloads/extracts if missing, and loads the Keras Datasets.
    
    Returns: (train_ds, val_ds, class_names) or (None, None, None) on failure.
    """
    # Check if the final structured data is present
    if not os.path.exists(DATA_DIR_RESIZED) or not os.listdir(DATA_DIR_RESIZED):
        print(f"Dataset not found at: {DATA_DIR_RESIZED}. Starting download/extraction process.")
        
        if not download_data():
            print("Aborting data preparation due to download failure.")
            return None, None, None
            
        if not extract_data():
            print("Aborting data preparation due to extraction failure.")
            return None, None, None
    else:
        print(f"Dataset already found and structured at: {DATA_DIR_RESIZED}. Skipping download/extraction.")
        
    # --- Data Loading Configuration uses global constants now ---
    
    # Check if real TF is available before attempting to load datasets
    if not hasattr(tf.keras.utils, 'image_dataset_from_directory'):
        print("Mock: Skipping Keras data loading as TensorFlow is not installed.")
        # Return mock datasets and mock class names for the mock training function
        mock_ds, _ = tf.keras.utils.MockKerasUtils().image_dataset_from_directory()
        return mock_ds, mock_ds, mock_ds.class_names
    
    # --- Try/Except Block for Keras Data Loading ---
    try:
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
            subset="validation",
            seed=SEED,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='categorical'
        )

        # --- CRITICAL FIX: Extract class names BEFORE performance optimization ---
        class_names = []
        if hasattr(train_ds, 'class_names'):
            class_names = train_ds.class_names
            print(f"\nInferred Class Names: {[c.lower() for c in class_names]}")
        else:
             print("Error: train_ds object is missing the 'class_names' attribute right after creation.")
             return None, None, None # Failure after creation

        print("\nData loading complete.")
        
        # --- Performance Optimization (These steps remove the class_names attribute) ---
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        # Return the new, optimized datasets *and* the class names list
        return train_ds, val_ds, class_names

    except Exception as e:
        # Catch any failure during the Keras setup process
        print(f"FATAL ERROR during Keras dataset creation: {e}")
        return None, None, None


if __name__ == '__main__':
    # Run data preparation directly to check if the data pipeline is working
    if load_and_prepare_data() != (None, None, None):
        print("Data preparation successful!")
    else:
        print("Data preparation failed.")

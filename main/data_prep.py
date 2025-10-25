import os
import sys
import zipfile
import requests
import shutil
import io # Needed for nested zip handling

# --- Configuration (Global Constants) ---
# NOTE: Replace 'YOUR_GITHUB_USERNAME' with your actual username after forking.
GITHUB_ZIP_URL = "https://github.com/bitresearch2006/trashnet/archive/refs/heads/master.zip"

DATA_DIR_BASE = 'trashnet_data'
DATA_DIR_RESIZED = os.path.join(DATA_DIR_BASE, 'dataset-resized')
ZIP_FILENAME = os.path.join(DATA_DIR_BASE, 'trashnet-master.zip')
SEED = 42 # Used for reproducibility

# Image and Data Loading Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

# Class names of the dataset (6 classes)
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
    import numpy as np # Needed for image handling in predict.py
    from tensorflow.keras.utils import image_dataset_from_directory
    
    print("TensorFlow data utilities available for use.")

except ImportError:
    # If TensorFlow/NumPy is not installed, define a robust mock
    print("Warning: TensorFlow and NumPy not found. Using mock class/functions.")

    class MockTF:
        """Mock object to simulate tensorflow package with necessary sub-modules."""
        class random:
            """Mock for tf.random to allow set_seed call."""
            def set_seed(self, seed):
                print(f"Mock: tf.random.set_seed called with {seed}")

        def image_dataset_from_directory(self, directory, labels, validation_split, subset, seed, image_size, batch_size, label_mode, **kwargs):
            """Mock function to simulate data loading and return mock data structures."""
            if subset == 'training':
                num_files = 2023
                subset_name = "Training"
            else:
                num_files = 505
                subset_name = "Validation"
                
            print(f"\n--- Loading {subset_name} Data ---")
            print(f"Mocking data loading from directory: {directory}...")
            
            # Simulate the classes found by the real function
            inferred_classes = [c.lower() for c in MOCK_CATEGORIES]
            
            # The previous log indicated 7 files found before cleanup, 
            # but after cleanup, it should be 6, so we ensure the mock reflects clean data.
            print(f"Found 2528 files belonging to {len(inferred_classes)} classes.")
            print(f"Using {num_files} files for {subset_name.lower()}.")
            
            # Simulate the output structure
            class MockDataset:
                """Mock dataset object."""
                element_spec = None
                class_names = inferred_classes
            
            return MockDataset()

    def image_dataset_from_directory(*args, **kwargs):
        """Wrapper for the MockTF method."""
        mock_tf_instance = MockTF()
        return mock_tf_instance.image_dataset_from_directory(*args, **kwargs)

    # Define the global 'tf' object as a mock when import fails
    tf = MockTF()
    np = type('MockNumpy', (object,), {'array': lambda x: x, 'expand_dims': lambda x, axis: x})() # Simple mock for numpy


# --- Data Download and Extraction ---

def cleanup_extracted_data():
    """Removes unwanted folders (like __MACOSX) and files from the extracted directory."""
    print("Starting data cleanup...")
    # ... (Cleanup logic remains the same) ...
    try:
        data_path = DATA_DIR_RESIZED
        # List items in the dataset directory
        for item in os.listdir(data_path):
            full_path = os.path.join(data_path, item)
            
            # Remove __MACOSX (and other hidden files/dirs starting with .)
            if item.startswith('__MACOSX') or item.startswith('.'):
                if os.path.isdir(full_path):
                    print(f"Removing extraneous directory: {item}")
                    shutil.rmtree(full_path)
                elif os.path.isfile(full_path):
                    print(f"Removing extraneous file: {item}")
                    os.remove(full_path)
        print("Data cleanup complete.")
    except Exception as e:
        print(f"Error during cleanup: {e}")


def download_and_extract_data():
    """
    Downloads the zipped dataset from GitHub, handles nested zips,
    and extracts the content to DATA_DIR_RESIZED.
    """
    print(f"Dataset not found at {DATA_DIR_RESIZED}. Starting download from GitHub...")

    # 1. Create base directory
    os.makedirs(DATA_DIR_BASE, exist_ok=True)

    # 2. Download the main ZIP file
    try:
        response = requests.get(GITHUB_ZIP_URL, stream=True)
        response.raise_for_status()
        with open(ZIP_FILENAME, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download successful. Starting extraction of main archive...")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {e}")
        return False

    # 3. Handle Nested ZIP Extraction
    try:
        with zipfile.ZipFile(ZIP_FILENAME, 'r') as outer_zip:
            # Look for the embedded dataset-resized.zip file
            embedded_zip_name = None
            for name in outer_zip.namelist():
                if name.endswith('dataset-resized.zip'):
                    embedded_zip_name = name
                    break

            if not embedded_zip_name:
                print("Error: 'dataset-resized.zip' not found inside the main GitHub ZIP.")
                return False

            print(f"Found embedded data archive: {embedded_zip_name}. Reading its contents...")
            
            # Extract the embedded zip file to a temporary stream/location
            with outer_zip.open(embedded_zip_name) as embedded_zip_file:
                # Read the bytes of the embedded zip file
                embedded_zip_bytes = embedded_zip_file.read()

                # Open the embedded zip file from the bytes
                with zipfile.ZipFile(io.BytesIO(embedded_zip_bytes), 'r') as inner_zip:
                    # Create the target directory
                    os.makedirs(DATA_DIR_RESIZED, exist_ok=True)
                    
                    # Extract contents of the inner zip to the target directory
                    inner_zip.extractall(DATA_DIR_RESIZED)

        print(f"Extraction complete. Dataset is ready at {DATA_DIR_RESIZED}")
        
        # 4. Perform cleanup (remove __MACOSX and other temporary files)
        cleanup_extracted_data()

        # 5. Clean up temporary files
        os.remove(ZIP_FILENAME)

        return True

    except Exception as e:
        print(f"Error during extraction: {e}")
        return False


def load_and_prepare_data():
    """
    Ensures data is present, then uses TensorFlow utilities (or mocks)
    to create training and validation datasets.
    """
    # Check if data exists; if not, download it
    if not os.path.exists(DATA_DIR_RESIZED) or not os.listdir(DATA_DIR_RESIZED):
        if not download_and_extract_data():
            return None, None
    else:
        print(f"Dataset already found and structured at: {DATA_DIR_RESIZED}. Skipping download.")
        
        # Ensure cleanup runs even if data was already present, just in case
        cleanup_extracted_data()

    # --- Data Loading Configuration uses global constants now ---
    
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

    # Print inferred class names (useful for debugging)
    if hasattr(train_ds, 'class_names'):
        print(f"\nInferred Class Names: {[c.lower() for c in train_ds.class_names]}")

    print("\nData loading complete. Ready for model training.")
    return train_ds, val_ds
    
# Allow direct execution to confirm data presence
if __name__ == "__main__":
    print("Attempting to ensure data is present for use in other scripts.")
    load_and_prepare_data()

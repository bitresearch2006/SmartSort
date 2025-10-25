import os
import sys
import json
import base64
import io
import logging
from data_prep import IMAGE_SIZE, MOCK_CATEGORIES, tf, np
from log_config import initialize_logger # Import the function

# --- Configuration ---

# **CRITICAL FIX: Determine the absolute path of the model file**
# This ensures the model is found even when executed from a different directory (like test/).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE_NAME = 'trashnet_classifier.keras'
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILE_NAME)

TEST_IMAGE_PATH = None  # Leave as None to auto-select a random image
TARGET_IMAGE_FORMAT = IMAGE_SIZE + (3,) # (224, 224, 3)

# Check if we are running the real TensorFlow environment
IS_REAL_TF = hasattr(tf, 'version')

if IS_REAL_TF:
    # Only import necessary Keras components if the real TF is available
    from tensorflow.keras.models import load_model
    # We use PIL's Image module to handle Base64 decoding and resizing
    from PIL import Image
    
else:
    # Define a robust mock for the Image module if not running real TF
    class MockImage:
        @staticmethod
        def open(fp):
            class MockImg:
                def resize(self, size): return self
                def getdata(self): return np.zeros(IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3) # Mock pixel data
            return MockImg()
    Image = MockImage

# --- Logging Setup ---
# Initialize and get the configured logger instance from the external module
# We use the full module path (__name__) for the logger name for clarity
logger = initialize_logger(log_name=__name__)


# --- Image Preprocessing ---

def preprocess_b64_image(image_b64: str):
    """
    Decodes a Base64 string, converts it to an image, resizes it,
    and converts it to a NumPy array for model prediction.
    """
    logger.info("Attempting to decode Base64 image and preprocess...")
    
    if not image_b64:
        logger.error("Input image_b64 string is empty.")
        return None
    
    try:
        # 1. Decode Base64 string to bytes
        image_bytes = base64.b64decode(image_b64)
        
        # 2. Open image from bytes stream (using PIL/MockImage)
        img = Image.open(io.BytesIO(image_bytes))
        
        # 3. Resize to model's required input size
        img = img.resize(IMAGE_SIZE)
        
        # Convert to NumPy array
        img_array = np.array(img, dtype='float32')
        
        # Check if we have 3 color channels (RGB)
        if img_array.ndim == 2: # Grayscale image
            logger.warning("Image is grayscale. Expanding to 3 channels.")
            img_array = np.stack([img_array]*3, axis=-1)
        elif img_array.shape[-1] == 4: # RGBA image
            logger.warning("Image has alpha channel. Removing it.")
            img_array = img_array[..., :3]

        # Expand dimensions to create a batch size of 1
        # Shape: (224, 224, 3) -> (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        logger.info(f"Image preprocessed successfully. Array shape: {img_array.shape}")
        return img_array.astype('float32')
        
    except Exception as e:
        logger.error(f"Error during image decoding/preprocessing: {e}")
        return None

# --- Main Prediction Function ---

# Load model globally once to save time on subsequent calls
_model = None

def get_model():
    """Loads the model once and caches it."""
    global _model
    if _model is not None:
        return _model
        
    # Check using the absolute path
    if not os.path.exists(MODEL_PATH):
        logger.critical(f"Model file not found at expected path: '{MODEL_PATH}'. Prediction cannot proceed.")
        return None

    if IS_REAL_TF:
        try:
            # Load the model (including custom augmentation layers)
            _model = load_model(MODEL_PATH)
            logger.info("Keras model loaded successfully.")
        except Exception as e:
            logger.critical(f"Error loading Keras model: {e}")
            _model = None
    else:
        logger.info(f"Mock: Loading mock model from {MODEL_PATH}")
        # Mock prediction function
        def mock_predict(*args, **kwargs):
            # Simulate a reasonable probability distribution based on the 6 classes
            return np.array([[0.3247, 0.1062, 0.1287, 0.0723, 0.0135, 0.3546]])

        class MockModel:
            def predict(self, *args, **kwargs):
                return mock_predict(*args, **kwargs)
        
        _model = MockModel()

    return _model

def handle(image_b64: str) -> str:
    """
    Classifies a waste image provided as a Base64 string and returns a JSON result string.
    
    Args:
        image_b64: The Base64 encoded string of the image.
        
    Returns:
        A JSON string containing the classification result and probability breakdown.
    """
    logger.info("--- Starting Classification Request ---")
    model = get_model()
    
    if model is None:
        return json.dumps({
            "status": "error",
            "message": "Model not loaded. Check server logs for details."
        })

    # 1. Preprocess the image from Base64
    processed_img = preprocess_b64_image(image_b64)
    if processed_img is None:
        return json.dumps({
            "status": "error",
            "message": "Failed to decode or preprocess image."
        })

    # 2. Predict
    try:
        # Note: verbose=0 suppresses the training progress bar
        predictions = model.predict(processed_img, verbose=0)
        
        # 3. Process results
        probabilities = predictions[0]
        predicted_index = np.argmax(probabilities)
        
        result_breakdown = {}
        for i, name in enumerate(MOCK_CATEGORIES):
            # FIX: Convert np.float32 to standard Python float before placing in dictionary
            # The float() casting resolves the JSON serialization error.
            result_breakdown[name] = float(round(probabilities[i] * 100, 2))
        
        predicted_class = MOCK_CATEGORIES[predicted_index]
        # FIX: Convert the confidence value to a standard Python float as well
        confidence = float(result_breakdown[predicted_class])

        # 4. Pack results into a single JSON object
        result_json = {
            "status": "success",
            "classification": predicted_class,
            "confidence_percent": confidence,
            "probability_breakdown": result_breakdown
        }
        
        logger.info(f"Classification successful: {predicted_class} with {confidence:.2f}% confidence.")
        logger.debug(f"Full breakdown: {result_breakdown}")
        
        return json.dumps(result_json)
            
    except Exception as e:
        logger.error(f"An unexpected error occurred during model prediction: {e}")
        # Note: We must ensure 'e' is also serializable, but usually str(e) is safe.
        return json.dumps({
            "status": "error",
            "message": f"Prediction failed due to internal error: {str(e)}"
        })

if __name__ == "__main__":
    # Example usage for local testing (requires a dummy base64 string)
    logger.info("Running script in local test mode.")
    
    # NOTE: In a real environment, you would replace this with a real base64 image string.
    # This is a very short, invalid placeholder string for demonstration.
    DUMMY_B64_IMAGE = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    
    # Run the handler and print the final JSON result
    result = handle(DUMMY_B64_IMAGE)
    
    # Pretty print the JSON output for readability
    print("\n--- API Response JSON ---")
    print(json.dumps(json.loads(result), indent=4))
    print("-------------------------")

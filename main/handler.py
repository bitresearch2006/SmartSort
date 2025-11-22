import os
import sys
import json
import base64
import io
import logging
import numpy as np
# Use modular configuration instead of data_prep dependency
from config_loader import load_categories, get_image_size, get_model_config
from log_config import initialize_logger # Import the function

# Initialize the logger at the start
logger = initialize_logger()

# --- Configuration ---
# **CRITICAL FIX: Determine the absolute path of the model file**
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load configuration from JSON files
CATEGORIES = load_categories()
IMAGE_SIZE = get_image_size()
MODEL_CONFIG = get_model_config()

# Model configuration
MODEL_FILE_NAME = MODEL_CONFIG.get('name', 'biomedical_waste_classifier.keras')
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILE_NAME)
TARGET_IMAGE_FORMAT = IMAGE_SIZE + (3,) # (224, 224, 3)

# Try to import TensorFlow and check availability
try:
    import tensorflow as tf
    IS_REAL_TF = hasattr(tf, 'version')
except ImportError:
    tf = None
    IS_REAL_TF = False

# Global variable to hold the loaded model
model = None

if IS_REAL_TF:
    # Only import necessary Keras components if the real TF is available
    try:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing import image as keras_image_processing
        from PIL import Image
        logger.info("TensorFlow utilities imported successfully for prediction.")
        
        # Pre-load the model globally to avoid loading it on every request (crucial for FaaS performance)
        model = None
        try:
            if os.path.exists(MODEL_PATH):
                model = load_model(MODEL_PATH)
                logger.info(f"Model successfully loaded from: {MODEL_PATH}")
            else:
                logger.error(f"MODEL NOT FOUND at {MODEL_PATH}. Prediction will fail.")
        except Exception as e:
            logger.error(f"Failed to load model from {MODEL_PATH}: {e}")
            model = None
            
    except ImportError as e:
        logger.error(f"Failed to import TensorFlow components: {e}")
        IS_REAL_TF = False
        model = None
        
if not IS_REAL_TF:
    # Define a robust mock for the Image module if not running real TF
    class MockImage:
        @staticmethod
        def open(fp):
            class MockImg:
                def resize(self, size): return self
                def getdata(self): return np.zeros(IMAGE_SIZE + (3,))
                def convert(self, mode): return self
            return MockImg()
    
    Image = MockImage # Use the mock class
    model = None
    logger.warning("Warning: Real TensorFlow libraries not found. Using mock utilities for prediction.")


# --- Prediction Helper Functions ---

def load_and_preprocess_image(image_b64):
    """
    Decodes a base64 string, loads it into a PIL Image object, 
    resizes it, and converts it to a NumPy array for prediction.
    """
    if not image_b64:
        logger.error("No image data provided.")
        return None

    try:
        # Decode the base64 string
        image_bytes = base64.b64decode(image_b64)
        image_stream = io.BytesIO(image_bytes)
        
        # Open and resize the image
        img = Image.open(image_stream).convert('RGB')
        img = img.resize(IMAGE_SIZE)

        # Convert to NumPy array - keep original [0, 255] range
        # The model's Rescaling layer will handle normalization to [-1, 1]
        img_array = np.array(img, dtype=np.float32)
        
        # Add batch dimension (224, 224, 3) -> (1, 224, 224, 3)
        processed_img = np.expand_dims(img_array, axis=0)

        logger.debug("Image decoded and preprocessed successfully.")
        return processed_img

    except Exception as e:
        logger.error(f"Error during image loading/preprocessing: {e}")
        return None

# --- Main FaaS Handler ---

def handle(req):
    """
    Handles the incoming HTTP request (JSON body containing base64 image).
    
    Args:
        req (str): JSON string containing the base64 image data under the key 'image_b64'.
        
    Returns:
        str: JSON string containing the classification result or an error message.
    """
    
    # 1. Check for Model Readiness
    if IS_REAL_TF and model is None:
        return json.dumps({
            "status": "error",
            "message": "Model not loaded. Check model path and TensorFlow installation."
        })
    
    try:
        # 2. Parse the Request
        input_data = json.loads(req)
        image_b64 = input_data.get("image_b64")

        if not image_b64:
            logger.warning("Received request with no 'image_b64' key.")
            return json.dumps({
                "status": "error",
                "message": "Missing 'image_b64' field in the request body."
            })
            
    except json.JSONDecodeError:
        logger.error("Failed to decode input JSON string.")
        return json.dumps({
            "status": "error",
            "message": "Invalid JSON input."
        })
    except Exception as e:
        logger.error(f"Error during request parsing: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Error parsing request: {str(e)}"
        })
        
    # 3. Preprocess Image
    processed_img = load_and_preprocess_image(image_b64)
    if processed_img is None:
        # load_and_preprocess_image logs the error.
        return json.dumps({
            "status": "error",
            "message": "Failed to decode or preprocess image data."
        })

    try:
        # 4. Predict
        if IS_REAL_TF:
            predictions = model.predict(processed_img, verbose=0)
        else:
            # Mock prediction (returns dummy probabilities, e.g., highest chance for the 2nd class)
            num_classes = len(CATEGORIES)
            mock_probs = [0.0] * num_classes
            mock_probs[1] = 0.95 # Mock prediction for the second class
            predictions = np.array([mock_probs], dtype=np.float32) # Ensure mock output is also float32

        # 5. Process results
        predicted_index = np.argmax(predictions[0])
        
        # Use the loaded CATEGORIES for class names
        predicted_class = CATEGORIES[predicted_index] 
        # Convert confidence to standard float before formatting
        confidence = float(predictions[0][predicted_index]) * 100

        # Create the probability breakdown for the JSON response
        result_breakdown = {}
        for name, prob in zip(CATEGORIES, predictions[0]):
            # CRITICAL FIX: Explicitly cast np.float32 (prob) to standard Python float
            result_breakdown[name] = float(f"{float(prob)*100:.2f}")

        # 6. Format JSON Response
        result_json = {
            "status": "success",
            "classification": predicted_class,
            "confidence_percent": confidence,
            "probability_breakdown": result_breakdown
        }
        
        logger.info(f"Classification successful: {predicted_class} with {confidence:.2f}% confidence.")
        logger.debug(f"Full breakdown: {result_breakdown}")
        
        # This will now succeed because all numerical values are standard Python types
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
    result = handle(json.dumps({"image_b64": DUMMY_B64_IMAGE}))
    print("Test Result:", result)
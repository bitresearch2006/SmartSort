import base64
import os
import sys
import requests
import json

# Add the path to the 'main' directory so we can import handler.py
# This assumes handler.py is in the parent directory of TestApp.py's directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'main')))

# Conditional import to prevent errors if running outside the structure
try:
    from handler import handle 
except ImportError:
    print("Error: Could not import 'handle' function. Ensure handler.py is accessible via the path setting.")
    sys.exit(1)


def remote_test(image_b64):
    """
    Simulates calling the deployed OpenFaaS function remotely.
    NOTE: Update the URL if deploying to a different location.
    """
    print("\n--- Running Remote Test (Requires OpenFaaS Deployment) ---")
    
    # Create the sub_json object with image data
    image_json = {"image_b64": image_b64}
    
    # Write JSON text to a file (optional, for debugging payload)
    json_text = json.dumps(image_json)              
    with open("request_payload.json", "w") as f:
        f.write(json_text)
    print("Request JSON payload written to request_payload.json")
    
    try:
        # Call OpenFaaS function (using the biomedical function name, if changed)
        # Assuming the function name is still 'ocr-detect' or a new one like 'biomedical-classifier'
        # Change the function name if you updated it in the OpenFaaS configuration.
        res = requests.post(
            "http://localhost:8080/function/biomedical-classifier", 
            json=image_json,
            timeout=10 # Add a timeout for robustness
        )

        print("\nRemote Function Response:")
        print(f"Status Code: {res.status_code}")
        
        # Pretty print the JSON response
        if res.status_code == 200:
            print("Response Body:")
            print(json.dumps(res.json(), indent=2))
        else:
            print("Response Body (Error):")
            print(res.text)

    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to the remote function.")
        print("Please ensure OpenFaaS is running and the function 'biomedical-classifier' is deployed.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during remote testing: {e}")


def local_test(image_b64):
    """
    Calls the handler.py function directly in the local environment for fast testing.
    """
    print("\n--- Running Local Test (Direct Call to handler.py) ---")
    # The handler expects a JSON string, so we must wrap the base64 string
    json_payload = json.dumps({"image_b64": image_b64})
    
    # Call the classification function directly
    result_json_string = handle(json_payload)

    # Parse and pretty print the result
    try:
        result = json.loads(result_json_string)
        print("Local Function Returned:")
        print(json.dumps(result, indent=2))
    except json.JSONDecodeError:
        print("Error: Handler returned invalid JSON.")
        print("Raw output:", result_json_string)


# --- Main Execution ---

# **UPDATE THE IMAGE FILE NAME HERE**
# Use a file name that suggests one of the new categories (e.g., 'syringeneedle.jpg')
# You MUST place a biomedical image file in the same directory as TestApp.py.
IMAGE_FILE = "(PE) Plastic equipment-packaging_IMG_6341_JPG.rf.0b8b9757cb4599d32b33bf770ea53dd9.jpg"  # Placeholder name for any biomedical waste image

# Determine whether to run local or remote test
if len(sys.argv) > 1 and sys.argv[1].lower() == 'remote':
    MODE = 'remote'
else:
    MODE = 'local'

if not os.path.exists(IMAGE_FILE):
    print(f"\nERROR: Test image file not found: '{IMAGE_FILE}'")
    print("Please place an image file (e.g., a syringe or glove) in this directory and update the IMAGE_FILE variable if needed.")
    sys.exit(1)

# Load image and convert to base64
print(f"Loading image: {IMAGE_FILE}")
with open(IMAGE_FILE, "rb") as image_file:
    image_b64 = base64.b64encode(image_file.read()).decode('utf-8')
    print(f"Image loaded and encoded ({len(image_b64)} bytes).")

# Run the selected test mode
if MODE == 'remote':
    remote_test(image_b64)
else:
    local_test(image_b64)
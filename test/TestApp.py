import base64
import os
import sys
import requests
import json

# Add the path to the 'main' directory so we can import predict.py's handle function
# Assuming TestApp.py is in a testing directory and predict.py is in 'main'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'main')))

# We expect the core handler function to be in 'predict.py'
try:
    from handler import handle
except ImportError:
    print("Error: Could not import 'handle' function from 'predict.py'.")
    print("Ensure 'predict.py' is in the 'main' directory and is correctly named.")
    sys.exit(1)


def remote_test(image_b64):
    """Sends the image to the remote OpenFaaS function for prediction."""
    print("\n--- Running Remote Test (OpenFaaS) ---")
    
    # NOTE: The OpenFaaS function expects the base64 image data wrapped in a JSON object.
    image_json = {"image_b64": image_b64}
    
    try:
        # Call OpenFaaS function (Update URL if required)
        res = requests.post(
            "http://localhost:8080/function/ocr-detect",
            json=image_json,
            timeout=30 # Set a reasonable timeout
        )
        
        # Print the raw response text to the terminal
        print(f"Remote Status Code: {res.status_code}")
        print("Remote Function Response:")
        # Attempt to pretty print if it's JSON
        try:
            print(json.dumps(res.json(), indent=4))
        except json.JSONDecodeError:
            print(res.text)

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to connect to remote function at http://localhost:8080/function/ocr-detect")
        print(f"Details: {e}")
        print("Is your OpenFaaS gateway running?")


def local_test(image_b64):
    """Calls the local 'handle' function directly for prediction."""
    print("\n--- Running Local Test (Direct Function Call) ---")

    # Optional: set env vars for logging in the local 'handle' call
    # These will be read by log_config.py/predict.py
    os.environ["DIAGNOSTICS"] = "true"
    os.environ["LOG_FILE"] = "SmartSort.log"
    
    # The 'handle' function returns a JSON string
    result_json_str = handle(image_b64)

    # Print the result to the terminal
    print("Local Function Result:")
    try:
        # Pretty print the JSON string output
        result_dict = json.loads(result_json_str)
        print(json.dumps(result_dict, indent=4))
    except json.JSONDecodeError:
        print(f"Function returned non-JSON data: {result_json_str}")


def main_loop():
    """Main loop to handle user input and execute tests."""
    
    # --- Setup ---
    # The image file to test with
    IMAGE_FILE = "glass17.jpg" 
    
    if not os.path.exists(IMAGE_FILE):
        print(f"CRITICAL ERROR: Test image '{IMAGE_FILE}' not found.")
        print("Please ensure this image is in the same directory as TestApp.py.")
        return

    # Load image and convert to base64 once outside the loop
    with open(IMAGE_FILE, "rb") as image_file:
        image_data = image_file.read()
        # Decode to utf-8 string as required by the 'handle' function
        image_b64 = base64.b64encode(image_data).decode("utf-8")

    
    while True:
        print("\n" + "="*50)
        print(f"Test Image: {IMAGE_FILE}")
        print("Which test would you like to perform?")
        choice = input("Enter [L]ocal, [R]emote, or [E]xit: ").strip().lower()

        if choice == 'e':
            print("Exiting Test Harness. Goodbye!")
            break
        elif choice == 'l':
            local_test(image_b64)
        elif choice == 'r':
            remote_test(image_b64)
        else:
            print("Invalid choice. Please enter 'L', 'R', or 'E'.")

if __name__ == "__main__":
    main_loop()

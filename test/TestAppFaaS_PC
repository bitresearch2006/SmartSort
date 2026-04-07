#!/usr/bin/env python3
import base64
import json
import requests

SMARTSORT_URL = "http://localhost:8080/function/SmartSort"

def test_remote(image_path):
    print("Reading image...")
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    payload = {
        "image": base64.b64encode(img_bytes).decode("utf-8")
    }

    print("Sending to SmartSort FAAS...")
    response = requests.post(SMARTSORT_URL, json=payload)
    print("Status:", response.status_code)
    print("Response:", response.text)

if __name__ == "__main__":
    # Modify to your local filename
    test_remote("Syringe_IMG_6408.JPG")

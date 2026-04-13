#!/usr/bin/env python3

import socket
import ssl
import json
import base64

HOST = "bitproduction.bitone.in"
PORT = 443
FUNCTION_NAME = "SmartSort"


def map_category_to_bin(category):
    category = category.lower().strip()

    if "metal" in category or "tweezer" in category:
        return 1
    elif "paper" in category:
        return 2
    elif "body" in category or "organic" in category or "gauze" in category:
        return 3
    elif "plastic" in category or "glove" in category or "mask" in category:
        return 4
    elif "needle" in category or "syringe" in category:
        return 5

    print("⚠️ Unknown category → default Bin 3")
    return 3


def send_to_faas(image_bytes):
    payload = {
        "image_b64": base64.b64encode(image_bytes).decode("utf-8")
    }

    body = json.dumps(payload)

    request = (
        f"POST /function/{FUNCTION_NAME} HTTP/1.1\r\n"
        f"Host: {HOST}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n\r\n"
        f"{body}"
    )

    sock = socket.create_connection((HOST, PORT))
    context = ssl._create_unverified_context()
    ssock = context.wrap_socket(sock, server_hostname=HOST)

    ssock.sendall(request.encode())

    response = b""
    while True:
        data = ssock.recv(4096)
        if not data:
            break
        response += data

    ssock.close()

    response_text = response.decode(errors="ignore")
    _, body = response_text.split("\r\n\r\n", 1)

    print(f"📡 FAAS Response: {body}")

    try:
        result = json.loads(body)
        category = result.get("classification", "")
    except:
        category = body.strip()

    bin_number = map_category_to_bin(category)

    print(f"🧠 Category: {category} → Bin {bin_number}")

    return bin_number

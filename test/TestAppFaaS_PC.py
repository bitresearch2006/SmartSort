import socket
import ssl
import json
import base64

HOST = "localhost"          # For local OpenFaaS
PORT = 8080                 # Gateway port
FUNCTION_NAME = "SmartSort"

def send_image(image_path):
    # Read image file
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    # Base64 encode
    payload = {
        "image_b64": base64.b64encode(img_bytes).decode("utf-8")
    }
    body = json.dumps(payload)

    # Build HTTP request
    request = (
        f"POST /function/{FUNCTION_NAME} HTTP/1.1\r\n"
        f"Host: {HOST}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n"
        "\r\n"
        f"{body}"
    )

    # Create socket & send
    sock = socket.create_connection((HOST, PORT))
    sock.sendall(request.encode())

    # Receive response
    response = b""
    while True:
        data = sock.recv(4096)
        if not data:
            break
        response += data
    sock.close()

    # Separate headers & body
    response_text = response.decode(errors="ignore")
    _, body = response_text.split("\r\n\r\n", 1)

    # Print SmartSort result
    print("SmartSort Response:")
    print(body)


if __name__ == "__main__":
    send_image("Syringe_IMG_6408.JPG")

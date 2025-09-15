import json
import socket
import threading

SERVER_HOST = "0.tcp.jp.ngrok.io"
SERVER_PORT = 12477

# Basic JSON-RPC message framing
def make_lsp_message(method, params, msg_id=1):
    body = json.dumps({
        "jsonrpc": "2.0",
        "id": msg_id,
        "method": method,
        "params": params
    })
    content_length = len(body.encode("utf-8"))
    return f"Content-Length: {content_length}\r\n\r\n{body}".encode("utf-8")

def read_responses(sock):
    buffer = ""
    while True:
        data = sock.recv(4096).decode("utf-8")
        if not data:
            break
        buffer += data

        # Process complete messages
        while "\r\n\r\n" in buffer:
            header, rest = buffer.split("\r\n\r\n", 1)
            headers = {}
            for line in header.split("\r\n"):
                if ": " in line:
                    k, v = line.split(": ", 1)
                    headers[k.lower()] = v

            if "content-length" in headers:
                length = int(headers["content-length"])
                if len(rest) < length:
                    # Wait for full message
                    break
                body = rest[:length]
                rest = rest[length:]
                buffer = rest  # keep remaining data for next iteration
                try:
                    msg = json.loads(body)
                    print("Response:", json.dumps(msg, indent=2))
                except json.JSONDecodeError:
                    print("Invalid JSON:", body)
            else:
                buffer = rest  # No content length found, discard headers

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((SERVER_HOST, SERVER_PORT))

    # Start a thread to listen for responses
    threading.Thread(target=read_responses, args=(sock,), daemon=True).start()

    # --- Initialize LSP session ---
    init_message = make_lsp_message("initialize", {
        "processId": None,
        "rootUri": None,
        "capabilities": {}
    })
    sock.sendall(init_message)

    # --- Call your custom command "generateCode" ---
    prompt = "Write a function to add two numbers"
    params = [
        "file:///test.py",          # fake URI
        {"line": 0, "character": 0},# insert at line 0
        prompt
    ]
    command_message = make_lsp_message("workspace/executeCommand", {
        "command": "generateCode",
        "arguments": params
    }, msg_id=2)

    sock.sendall(command_message)

    input("Press Enter to exit...\n")
    sock.close()

if __name__ == "__main__":
    main()

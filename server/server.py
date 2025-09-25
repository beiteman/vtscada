# server.py
import socket
import json
import threading
import signal
import sys
from typing import Dict, Any, List
from dotenv import load_dotenv
import os
import re
import traceback
from vtscada_search import run_inference, FunctionDoc, Parameter

# -------------------------
# INFERENCE CONFIGURATION
# -------------------------
docs_path = "vtscada_functions.json"
vec_type = "tfidf"
stem = True
min_df = 1
max_df = 1.0
ngram = (1, 1)
name_weight = 3
clusters = None
model = "grid_results.pkl"
dense_model = "sentence-transformers/all-MiniLM-L6-v2"
dense_norm = True
dense_batch = 256
dense_device = "gpu"

def _query(prompt: str, top_k = 5) -> str:
    if os.getenv('ENVIRONMENT') == "TEST":
        return ["ReadLock", "SetShelved", "BuffWrite", "DLL", "SelectDAG"]
    else: 
        docs = run_inference(docs_path=docs_path, query=prompt,
                            query_file = None, top_k=top_k, vec_type=vec_type, stem=stem, 
                            min_df=min_df, max_df=max_df, ngram=ngram, name_weight=name_weight,
                            clusters=clusters, model=model, dense_model=dense_model,
                            dense_batch=dense_batch, dense_device=dense_device)
        return [doc.name for doc in docs]

with open("vtscada_functions.json", "r", encoding="utf-8") as f:
    vtscada_functions = {}
    for item in json.load(f):
        vtscada_functions[item["name"].lower()] = item

def _to_insert_text(function_name: str, leading_whitespaces: str = "") -> str:
    if not function_name:
        return None
    
    if function_name.lower() not in vtscada_functions:
        print(function_name, "is not exists")
        return None
    
    _function = vtscada_functions[function_name.lower()]
    
    result_lines = []
    result_lines.append("{ " + f"{_function['name']}: {_function['description']}" + " }")
    
    _snippets = _function["example"]
    if not _snippets:
        _snippets = _function["format"]
        
    if _snippets:
        _snippets = _snippets.strip()
        
    lines = _snippets.split("\r\n")
    for i, line in enumerate(lines):
        if i > 0: 
            result_lines.append(f"{leading_whitespaces}\t{line.strip()}")
        else: result_lines.append(f"{leading_whitespaces}{line.strip()}")
            
    return "\n" + "\n".join(result_lines).strip()

def generate_completions(request: Dict[str, Any]) -> Dict[str, Any]:
    document = request['document']
    position = request['position']
    
    character_pos = position['character']
    line_pos = position['line']
    
    text = document['text']
    lines = text.split('\n')
    line = lines[line_pos]
    
    line_trim = line.strip()
    leading_whitespaces = re.match(r"\s*", line).group()
    
    items = []
    if line_trim.startswith("{") and line_trim.endswith("}"):
        query = line_trim[1:-1].strip()
        if query:
            function_names = _query(query)
            for function_name in function_names:
                insert_text = _to_insert_text(function_name, leading_whitespaces)
                if insert_text:
                    items.append({
                        "label": function_name,
                        "insertText": insert_text,
                        "range": {
                            "start": {"line": line_pos, "character": character_pos+1},
                            "end": {"line": line_pos, "character": character_pos+1}
                        },
                        "command": {
                            "command": "vtscada.query",
                            "title": "Query result",
                            "arguments": []
                        }
                    })
    return {"items": items}

class TCPServer:
    def __init__(self, host='localhost', port=3000):
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False
        self.active_connections = []
        
        # Set up signal handling
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        self.shutdown()

    def handle_client(self, conn: socket.socket, addr: tuple):
        print(f"Connected by {addr}")
        self.active_connections.append(conn)
        try:
            while self.running:
                data = conn.recv(4096)
                if not data:
                    break
                
                try:
                    request = json.loads(data.decode('utf-8'))
                    response = generate_completions(request)
                    conn.sendall((json.dumps(response) + '\n').encode('utf-8'))
                except json.JSONDecodeError as e:
                    error_response = {"error": f"Invalid JSON: {str(e)}"}
                    conn.sendall((json.dumps(error_response) + '\n').encode('utf-8'))
                    
        except ConnectionResetError:
            print(f"Client {addr} disconnected unexpectedly")
        finally:
            conn.close()
            # Use a lock or a thread-safe list if multiple threads can modify active_connections at the same time
            if conn in self.active_connections:
                self.active_connections.remove(conn)

    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        
        # Set a timeout on the socket to make accept() non-blocking
        self.server_socket.settimeout(1.0) 
        
        self.server_socket.listen()
        self.running = True
        
        print(f"Server listening on {self.host}:{self.port}")
        try:
            while self.running:
                try:
                    conn, addr = self.server_socket.accept()
                    thread = threading.Thread(target=self.handle_client, args=(conn, addr))
                    thread.daemon = True  # Daemon threads will exit when main exits
                    thread.start()
                except socket.timeout:
                    # This is expected and allows the loop to check `self.running`
                    continue 
        except Exception as e:
            if self.running:
                print(f"Server error: {str(e)}")
        finally:
            self.shutdown()

    def shutdown(self):
        print("Shutting down server...")
        self.running = False
        
        # Close all active connections
        for conn in self.active_connections:
            try:
                conn.shutdown(socket.SHUT_RDWR)
                conn.close()
            except:
                pass
        
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        print("Server shutdown complete")
        sys.exit(0)


if __name__ == '__main__':
    load_dotenv()
    server_host = os.getenv('SERVER_HOST', '0.0.0.0')
    server_port = int(os.getenv('SERVER_PORT', '3000'))
    server = TCPServer(host=server_host, port=server_port)
    server.start()
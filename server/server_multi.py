import socket
import threading
from pygls.server import LanguageServer
from lsprotocol.types import WorkspaceEdit, TextEdit, Range, Position
from typing import List
from vtscada_search import run_inference

def create_ls():
    ls = LanguageServer("pygen-lsp", "0.0.1")

    @ls.command("generateCode")
    async def generate_code(server: LanguageServer, params: List[str]):
        try:
            if len(params) < 3:
                return "Missing arguments. Expect: uri, position, prompt."

            uri, position_dict, prompt = params[0], params[1], params[2]
            pos = Position(line=int(position_dict["line"]), character=int(position_dict["character"]))

            new_text = inference(prompt)
            text_edit = TextEdit(range=Range(start=pos, end=pos), new_text=new_text)

            we = WorkspaceEdit(changes={uri: [text_edit]})
            apply_resp = await server.apply_edit_async(we)

            if apply_resp.applied:
                return "Insertion applied."
            else:
                return f"Client refused edit: {apply_resp.failure_reason or 'unknown reason'}"
        except Exception as e:
            return f"generateCode failed: {str(e)}"

    return ls


# Your inference function
def inference(prompt: str, top_k=5) -> str:
    docs = run_inference(
        docs_path="vtscada_functions.json",
        query=prompt,
        query_file=None,
        top_k=top_k,
        vec_type="tfidf",
        stem=True,
        min_df=1,
        max_df=1.0,
        ngram=(1, 1),
        name_weight=3,
        clusters=None,
        model="grid_results.pkl",
        dense_model="sentence-transformers/all-MiniLM-L6-v2",
        dense_batch=256,
        dense_device="gpu"
    )

    result = ""
    for doc in docs:
        result += f"\n{{{doc.description}}}\n{doc.name}"
    return result + "\n"


def handle_client(conn, addr):
    print(f"Client connected: {addr}")
    ls = create_ls()
    ls.start_io(conn, conn)
    print(f"Client disconnected: {addr}")


if __name__ == "__main__":
    HOST, PORT = "0.0.0.0", 2087
    print(f"Server listening on {HOST}:{PORT}...")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()

        while True:
            conn, addr = s.accept()
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

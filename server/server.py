#!/usr/bin/env python
from pygls.server import LanguageServer
from lsprotocol.types import WorkspaceEdit, TextEdit, Range, Position
from typing import List
from vtscada_search import run_inference, FunctionDoc, Parameter

ls = LanguageServer("pygen-lsp", "0.0.1")

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

def inference(prompt: str, top_k = 5) -> str:
    print("------------PROMPT-------------")
    print(prompt)
    print("-------------------------------")
    docs = run_inference(docs_path=docs_path, query=prompt,
                        query_file = None, top_k=top_k, vec_type=vec_type, stem=stem, 
                        min_df=min_df, max_df=max_df, ngram=ngram, name_weight=name_weight,
                        clusters=clusters, model=model, dense_model=dense_model,
                        dense_batch=dense_batch, dense_device=dense_device)
    
    print(f"RESULT: {len(docs)}")
    result = ""
    for doc in docs:
        name = doc.name
        desc = doc.description
        desc_cmt = "{" + desc + "}"
        result = f"{result}\n{desc_cmt}\n{name}"
    return result + "\n"

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
        
        # Use apply_edit_async to await the client's response
        apply_resp = await server.apply_edit_async(we)

        if apply_resp.applied:
            return "Insertion applied."
        else:
            return f"Client refused edit: {apply_resp.failure_reason or 'unknown reason'}"
    except Exception as e:
        return f"generateCode failed: {str(e)}"

if __name__ == "__main__":
    print("Starting ...")
    ls.start_tcp("0.0.0.0", 2087)
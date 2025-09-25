from typing import List
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

def run(prompt: str, top_k = 5) -> str:
    docs = run_inference(docs_path=docs_path, query=prompt,
                        query_file = None, top_k=top_k, vec_type=vec_type, stem=stem, 
                        min_df=min_df, max_df=max_df, ngram=ngram, name_weight=name_weight,
                        clusters=clusters, model=model, dense_model=dense_model,
                        dense_batch=dense_batch, dense_device=dense_device)
    return [doc.name for doc in docs]
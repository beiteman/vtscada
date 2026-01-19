import joblib
import onnx
import numpy as np
import json
import re, unicodedata
import jieba
import scipy.sparse as sp

from onnx import helper, numpy_helper, TensorProto
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

from models.vtscada_search import run_inference, FunctionDoc, Parameter
from models.vtscada_search import VTScadaSearcher

def chinese_tokenizer(text):
    return list(jieba.cut(text))


# # --- Configuration ---
# PKL_PATH = Path("grid_results.pkl")
# ONNX_PATH = "tfidf_matrix.onnx"
# # ---------------------

# --- Document Text Function (copied from your original) ---
def doc_to_text(doc):
    name_weight = 3
    parts = []
    
    # Use getattr for safer attribute access on FunctionDoc/Parameter objects
    doc_name = getattr(doc, 'name', None)
    if doc_name:
        parts.extend([doc_name] * name_weight)
        
    for field_name in ['description', 'function_groups', 'related_to', 'comments', 'usage', 'returns']:
        field_value = getattr(doc, field_name, None)
        if field_value:
            parts.append(field_value)
            
    parameters = getattr(doc, 'parameters', [])
    for p in parameters:
        param_desc = getattr(p, 'description', None)
        if param_desc:
            parts.append(param_desc)
            
    combined = " ".join(parts).lower()
    return clean_text_model(combined)
    # Note: Using re.sub requires importing 're', which the minimal script might not have.
    # We will trust the original text cleanup was already done correctly.
    # return combined.replace(r"[^\w\s]", " ").replace(r"\s+", " ").strip()
# ------------------------------------------------------------------------------------------------

# 1. Load Data
# We must use load_model from your original script because it handles the searcher object
def load_searcher(pkl: Path) -> VTScadaSearcher:
    payload = joblib.load(pkl)
    # The keys 'vectorizer' and 'docs' are guaranteed to exist, but their values might be None
    
    # We must instantiate the searcher using the config saved in the pickle
    s = VTScadaSearcher(**{k: payload["cfg"][k] for k in payload["cfg"]})
    s.vectorizer = payload["vectorizer"]
    s.matrix = payload["matrix"]
    s.docs = payload["docs"]
    s.labels_ = payload["labels"]
    s.bm25 = payload.get("bm25")
    s.tok_docs = payload.get("tok_docs", [])
    s.dense_matrix = payload.get("dense_matrix")
    return s

def searcher_diag(pkl: Path):
    searcher = load_searcher(pkl)
    print("=== SEARCHER DIAGNOSTICS ===")
    print("searcher type:", type(searcher))
    print("cfg vectorizer_type:", getattr(searcher, "cfg", {}).get("vectorizer_type"))
    print("vectorizer object:", type(getattr(searcher, "vectorizer", None)))
    print("matrix type:", type(getattr(searcher, "matrix", None)))

    mat = getattr(searcher, "matrix", None)
    if mat is None:
        print("matrix is None")
    else:
        # If sparse
        if sp.issparse(mat):
            print("matrix is sparse:", mat.__class__)
            print("matrix shape:", mat.shape)
            print("matrix nnz (nonzeros):", mat.nnz)
            print("matrix density (nnz / total):", mat.nnz / (mat.shape[0] * mat.shape[1]))
            print("matrix dtype:", mat.dtype)
        else:
            print("matrix is dense numpy array")
            print("matrix shape:", getattr(mat, "shape", None))
            print("matrix nbytes (approx):", getattr(mat, "nbytes", None))
            # show some stats
            a = np.asarray(mat)
            print("dtype:", a.dtype, "min,max,mean:", a.min(), a.max(), a.mean())
    print("vocab type:", type(getattr(searcher, "vectorizer", None)))
    vocab = getattr(getattr(searcher, "vectorizer", None), "vocabulary_", None)
    print("vocab is None?", vocab is None)
    if vocab is not None:
        print("vocab len:", len(vocab))
        for i, (k, v) in enumerate(list(vocab.items())[:40]):
            print(i, repr(k), "->", v)
        # show any bytes keys
        bytes_keys = [k for k in vocab.keys() if isinstance(k, bytes)]
        print("num bytes keys:", len(bytes_keys))
    print("============================")


def model2onnx(pkl_path, onnx_path, model_info_path, vectorizer_type = "tfidf"):
    try:
        searcher = load_searcher(pkl_path)
    except Exception as e:
        print(f"FATAL: Could not load the full Searcher object. Ensure the original file 'vtscada_search.py' (or the relevant classes) is importable.")
        print(f"Error: {e}")
        exit()

    # 2. Check Vectorizer type and re-index if needed (THE KEY FIX)
    if searcher.vectorizer is None or searcher.matrix is None:
        print("\nWARNING: The best model in the pickle was not TFIDF/Count (vectorizer is None).")
        
        # To proceed with ONNX, we must use the docs and force a TFIDF vectorizer.
        # ASSUME the original grid search chose 'tfidf' and 'stemming=True' with min_df=1, max_df=1.0
        
        # Fallback: Re-instantiate the *best* sparse vectorizer based on the config
        cfg = searcher.cfg
        print(f"Re-indexing data using original config: {cfg['vectorizer_type']}...")
        
        # Create a new searcher with sparse settings
        sparse_searcher = VTScadaSearcher(
            vectorizer_type=vectorizer_type, # Force TFIDF for ONNX conversion
            use_stemming=cfg.get('use_stemming', True),
            min_df=cfg.get('min_df', 1),
            max_df=cfg.get('max_df', 1.0),
            ngram_range=cfg.get('ngram_range', (1, 1)),
            name_weight=cfg.get('name_weight', 3),
        )
        sparse_searcher.index(searcher.docs)
        
        vectorizer = sparse_searcher.vectorizer
        doc_matrix = sparse_searcher.matrix.toarray().astype(np.float32)
        docs = sparse_searcher.docs
        
    else:
        print("Found Vectorizer and Matrix from pickle. Proceeding with export.")
        vectorizer = searcher.vectorizer
        # Convert sparse matrix to dense NumPy array for ONNX
        doc_matrix = searcher.matrix.toarray().astype(np.float32) 
        docs = searcher.docs



    # 3. Validation
    if vectorizer is None:
        print("\nFATAL: Failed to create or retrieve a sparse vectorizer. Cannot proceed with ONNX export.")
        exit()

    # 4. Define the ONNX Graph
    input_name = 'query_vector'
    input_size = doc_matrix.shape[1] # Vocabulary size

    X = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [None, input_size])

    doc_matrix_name = 'doc_matrix'
    # Transposed for MatMul: [Vocab_Size, Num_Docs]
    doc_matrix_initializer = numpy_helper.from_array(doc_matrix.T, name=doc_matrix_name)

    output_name = 'similarity_scores'
    Y = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [None, doc_matrix.shape[0]])

    matmul_node = helper.make_node(
        'MatMul',
        [input_name, doc_matrix_name],
        [output_name],
        name='dot_product_node'
    )

    graph = helper.make_graph(
        [matmul_node],
        'tfidf_search_graph',
        [X],
        [Y],
        [doc_matrix_initializer]
    )

    # 5. Create and Save the Model and Info
    onnx_model = helper.make_model(graph, producer_name='vtscada_exporter')
    onnx.save(onnx_model, onnx_path)
    
    print(f"\n✅ ONNX model saved to {onnx_path}")
    print("num_docs:", doc_matrix.shape[0])
    print("vocab_size:", doc_matrix.shape[1])
    print("doc_matrix dtype:", doc_matrix.dtype)
    print("doc_matrix nbytes:", doc_matrix.nbytes)
    
    model_info = {
        "vocabulary": vectorizer.vocabulary_,
        "keys": [d.name for d in docs]
    }
    with open(model_info_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    print(f"✅ Model info (vocab, docs) saved to {model_info_path}")

def clean_text_model(s):
    if not s:
        return ""
    # Normalize and remove weird control characters
    s = unicodedata.normalize('NFC', s)
    # replace all non-word (unicode) characters with a space
    s = re.sub(r"[^\w\s]+", " ", s, flags=re.UNICODE)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def clean_text(text):
    essential_whitespace = ('\n', '\t', '\r')
    cleaned_chars = [
        char for char in text 
        if char.isprintable() or char in essential_whitespace
    ]
    return "".join(cleaned_chars)

from typing import List


def word_wrap(text: str, max_width: int = 70, indent: bool = True) -> List[str]:
    if not text:
        return []
    if max_width <= 0:
        return text.replace(' ', '\n') # Break every space if width is too small or zero

    words = text.split()
    if not words:
        return []

    wrapped_lines = []
    current_line_words = []
    current_line_length = 0

    for word in words:
        word_len = len(word)

        # Calculate potential length: current length + space (if line not empty) + word length
        potential_len = current_line_length + (1 if current_line_words else 0) + word_len

        if potential_len <= max_width:
            # The word fits on the current line
            current_line_words.append(word)
            # Update length for the next iteration (including the space added before this word)
            current_line_length = potential_len
        else:
            # The word does not fit, start a new line
            if current_line_words:
                wrapped_lines.append(" ".join(current_line_words))

            # Start the new line with the current word
            current_line_words = [word]
            current_line_length = word_len # Length of the new line is just the word length

            # Edge case: If a single word is longer than max_width, it will be on its own line.
            # You might use the standard library `textwrap` for more complex hyphenation logic.

    # Add the last remaining line
    if current_line_words:
        wrapped_lines.append(" ".join(current_line_words))

    if indent:
        _wrapped_lines = []
        for i, line in enumerate(wrapped_lines):
            if i==0: _wrapped_lines.append(line)
            else: _wrapped_lines.append(f"\t{line}")
        return _wrapped_lines
    else:
        return wrapped_lines

def doc_to_snippets(doc) -> List[str]:
    result = []
    if "format" in doc:
        if "(" in doc["format"]:
            index = doc["format"].index('(')
            fn_name = clean_text(doc["format"][:index].strip())
        else:
            fn_name = clean_text(doc["format"].strip())
        if doc.get("parameters"):
            result.append(f"{fn_name}(")
            for i, param in enumerate(doc["parameters"]):
                param_desc = clean_text(param["description"])
                if len(param_desc) > 100:
                    # select only two sentences
                    sentences = re.split(r'(?<=[.!?])\s+', param_desc.strip())
                    param_desc = " ".join(sentences[:2])
                    
                comment = "{ " + param_desc + " }" if param_desc.strip() else ""
                if i < (len(doc["parameters"]) - 1):
                    lines = word_wrap(f"\t{param['name']}, {comment}")
                    for line in lines:
                        result.append(f"\t{line}")
                else: # last param
                    lines = word_wrap(f"\t{param['name']} {comment}")
                    for line in lines:
                        result.append(f"\t{line}")
            result.append(");")
        else:
            result.append(f"{fn_name}();")
    
    return result
        
def doc_to_comments(doc, max_with_word_wrap = 70) -> List[str]:
    description = clean_text((doc.get("description") or "").strip())
    comments = clean_text((doc.get("comments") or "").strip())
    
    result = []
    if description:
        result += word_wrap("{ Description: " + description + " }", max_width=max_with_word_wrap)
    if comments:
        result += word_wrap("{ " + comments + " }", max_width=max_with_word_wrap)
    
    return result

# resources/info.${lang}.json
# resources/model.${lang}.json
# resources/docs.${lang}.json
import sys
if __name__ == "__main__":
    # searcher_diag(pkl="models/grid_results_zh-TW.pkl")
    # sys.exit(0)
    
    model2onnx(pkl_path="models/grid_results_en.pkl", 
               onnx_path="resources/model.en.onnx", 
               model_info_path="resources/info.en.json",
               vectorizer_type="tfidf")
    
    # model2onnx(pkl_path="models/grid_results_zh-TW.pkl", 
    #            onnx_path="resources/model.zh-tw.onnx", 
    #            model_info_path="resources/info.zh-tw.json",
    #            vectorizer_type="count")
    
    # model2onnx(pkl_path="models/grid_results_zh-CN.pkl", 
    #            onnx_path="resources/model.zh-cn.onnx", 
    #            model_info_path="resources/info.zh-cn.json",
    #            vectorizer_type="count")
    
    # ----------------------
    # prepare the docs
    # ----------------------
    with open("models/vtscada_functions_en.json", "r", encoding="utf-8") as f:
        func_en = json.load(f)
    
    with open("resources/docs.en.json", "w", encoding="utf-8") as f:
        docs_en = {}
        for doc in func_en:
            docs_en[doc["name"]] = {
                "comments": doc_to_comments(doc),
                "snippets": doc_to_snippets(doc),
                "is_steady_state": "steady" in (doc.get("usage") or "").lower(),
                "is_script": "script" in (doc.get("usage") or "").lower()
            }
        json.dump(docs_en, f, ensure_ascii=False, indent=4)
    
    with open("models/vtscada_functions_zh-TW.json", "r", encoding="utf-8") as f:
        func_tw = json.load(f)
        
    with open("resources/docs.zh-tw.json", "w", encoding="utf-8") as f:
        docs_tw = {}
        for doc in func_tw:
            name = doc["name"]
            docs_tw[name] = {"comments": doc_to_comments(doc, 30)}
            # extract other fields from en
            doc_en = docs_en.get(name)
            snippets, is_steady_state, is_script = [], False, False
            if doc_en:
                docs_tw[name]["snippets"] = doc_en["snippets"]
                docs_tw[name]["is_steady_state"] = doc_en["is_steady_state"]
                docs_tw[name]["is_script"] = doc_en["is_script"]
        json.dump(docs_tw, f, ensure_ascii=False, indent=4)
        
    with open("models/vtscada_functions_zh-CH.json", "r", encoding="utf-8") as f:
        func_cn = json.load(f)
    
    with open("resources/docs.zh-cn.json", "w", encoding="utf-8") as f:
        docs_cn = {}
        for doc in func_cn:
            name = doc["name"]
            docs_cn[name] = {"comments": doc_to_comments(doc, 30)}
            # extract other fields from en
            doc_en = docs_en.get(name)
            snippets, is_steady_state, is_script = [], False, False
            if doc_en:
                docs_cn[name]["snippets"] = doc_en["snippets"]
                docs_cn[name]["is_steady_state"] = doc_en["is_steady_state"]
                docs_cn[name]["is_script"] = doc_en["is_script"]
        json.dump(docs_cn, f, ensure_ascii=False, indent=4)
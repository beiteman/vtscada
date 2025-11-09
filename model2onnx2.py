#!/usr/bin/env python3
"""
export_pipeline.py

Rewritten pipeline that:
- preserves existing English flow
- fixes Chinese tokenization/cleaning when needed
- handles missing BM25/vectorizer by rebuilding a TF-IDF/Count index from docs
- diagnostics and safe ONNX export (auto float16 fallback)
"""

import joblib
import onnx
import numpy as np
import json
import re
import unicodedata
import os
import sys
import math
import scipy.sparse as sp

from pathlib import Path
from typing import List

# optional dependency for Chinese tokenization
try:
    import jieba
except Exception:
    jieba = None

from onnx import helper, numpy_helper, TensorProto
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# import your project searcher; keep as you had
from models.vtscada_search import run_inference, FunctionDoc, Parameter
from models.vtscada_search import VTScadaSearcher

# -------------------------
# Configurable parameters
# -------------------------
# If estimated initializer bytes exceed this, converter will auto-cast to float16.
ONNX_SIZE_WARNING_THRESHOLD_BYTES = 200 * 1024 * 1024  # 200 MB

# If you prefer to always use float16 for huge matrices, set True.
AUTO_FALLBACK_TO_FLOAT16 = True

# -------------------------
# Text cleaning / tokenization
# -------------------------
def clean_text_model(s: str) -> str:
    """Unicode-aware cleaning used for building vectorizer input.
       Keeps Chinese characters intact but removes punctuation and collapses whitespace."""
    if not s:
        return ""
    s = unicodedata.normalize('NFC', s)
    # Replace punctuation and other non-word/space with a space. Use UNICODE flag.
    s = re.sub(r"[^\w\s\u4e00-\u9fff]+", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def clean_text_keep_printable(text: str) -> str:
    """Light cleaning for doc->snippets/comments etc."""
    if not text:
        return ""
    return "".join(ch for ch in text if ch.isprintable() or ch in ("\n", "\t", "\r"))

def contains_cjk(text: str) -> bool:
    """Return True if `text` contains any CJK unified ideograph char."""
    for ch in text:
        code = ord(ch)
        # Basic CJK Unified Ideographs block: 0x4E00 - 0x9FFF
        if 0x4E00 <= code <= 0x9FFF:
            return True
    return False

def is_cjk_corpus(docs: List) -> bool:
    """Heuristic: check first N docs for CJK characters in their text."""
    n = min(200, max(10, len(docs)))
    for i, d in enumerate(docs[:n]):
        s = doc_to_text(d)
        if contains_cjk(s):
            return True
    return False

def doc_to_text(doc) -> str:
    """Aggregate fields of FunctionDoc/Parameter into a single cleaned string."""
    name_weight = 3
    parts = []
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
    combined = " ".join(parts).strip()
    return clean_text_model(combined)

def chinese_tokenizer(text: str):
    if not jieba:
        raise RuntimeError("jieba is required for Chinese tokenization. Install with `pip install jieba`.")
    # We return list of tokens for sklearn's tokenizer
    return list(jieba.cut(text))

# -------------------------
# Loader & diagnostics
# -------------------------
def load_searcher(pkl: Path) -> VTScadaSearcher:
    payload = joblib.load(pkl)
    # instantiate using cfg contents
    s = VTScadaSearcher(**{k: payload["cfg"][k] for k in payload["cfg"]})
    # restore possible saved fields
    s.vectorizer = payload.get("vectorizer")
    s.matrix = payload.get("matrix")
    s.docs = payload.get("docs")
    s.labels_ = payload.get("labels")
    s.bm25 = payload.get("bm25")
    s.tok_docs = payload.get("tok_docs", [])
    s.dense_matrix = payload.get("dense_matrix")
    s.cfg = payload.get("cfg", s.cfg if hasattr(s, "cfg") else {})
    return s

def print_searcher_diag(searcher: VTScadaSearcher):
    print("=== SEARCHER DIAGNOSTICS ===")
    print("searcher type:", type(searcher))
    print("cfg vectorizer_type:", getattr(searcher, "cfg", {}).get("vectorizer_type"))
    print("vectorizer object:", type(getattr(searcher, "vectorizer", None)))
    print("matrix type:", type(getattr(searcher, "matrix", None)))
    mat = getattr(searcher, "matrix", None)
    if mat is None:
        print("matrix is None")
    else:
        if sp.issparse(mat):
            print("matrix is sparse:", mat.__class__)
            print("matrix shape:", mat.shape)
            print("matrix nnz (nonzeros):", mat.nnz)
            print("matrix density (nnz / total):", mat.nnz / (mat.shape[0] * mat.shape[1]))
            print("matrix dtype:", mat.dtype)
        else:
            a = np.asarray(mat)
            print("matrix is dense numpy array")
            print("matrix shape:", a.shape)
            print("matrix nbytes (approx):", a.nbytes)
            print("dtype:", a.dtype, "min,max,mean:", a.min(), a.max(), a.mean())
    vocab = getattr(getattr(searcher, "vectorizer", None), "vocabulary_", None)
    print("vocab is None?", vocab is None)
    if vocab is not None:
        print("vocab len:", len(vocab))
        for i, (k, v) in enumerate(list(vocab.items())[:40]):
            print(i, repr(k), "->", v)
        bytes_keys = [k for k in vocab.keys() if isinstance(k, bytes)]
        print("num bytes keys:", len(bytes_keys))
    print("============================")

# -------------------------
# Rebuild vectorizer fallback
# -------------------------
def rebuild_vectorizer_from_docs(docs: List, vectorizer_type: str = "tfidf"):
    """Rebuild TF-IDF or Count vectorizer from raw docs with appropriate tokenizer for Chinese."""
    # prepare corpus: cleaned strings
    texts = [doc_to_text(d) for d in docs]
    # decide if the corpus is Chinese
    chinese = is_cjk_corpus(docs)
    print("Rebuild vectorizer: detected_chinese_corpus =", chinese, "vectorizer_type=", vectorizer_type)
    if chinese:
        # prefer jieba word analyzer; fallback to char n-grams if jieba is missing
        if jieba:
            if vectorizer_type.lower() == "count":
                vec = CountVectorizer(tokenizer=chinese_tokenizer, analyzer="word", min_df=1)
            else:
                vec = TfidfVectorizer(tokenizer=chinese_tokenizer, analyzer="word", min_df=1)
        else:
            # fallback to char n-gram (keeps pipeline working but less ideal segmentation)
            print("WARNING: jieba not installed; using char n-gram analyzer for Chinese text.")
            if vectorizer_type.lower() == "count":
                vec = CountVectorizer(analyzer="char", ngram_range=(1,2), min_df=1)
            else:
                vec = TfidfVectorizer(analyzer="char", ngram_range=(1,2), min_df=1)
    else:
        # English or Latin-based: use reasonable defaults
        if vectorizer_type.lower() == "count":
            vec = CountVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b", min_df=1)
        else:
            vec = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b", min_df=1)
    X = vec.fit_transform(texts)
    print("Rebuilt vectorizer: shape:", X.shape, "nonzeros:", X.nnz if sp.issparse(X) else np.count_nonzero(X))
    return vec, X, texts

# -------------------------
# ONNX exporter core
# -------------------------
def safe_create_initializer(doc_matrix: np.ndarray, name: str, auto_float16_threshold=ONNX_SIZE_WARNING_THRESHOLD_BYTES):
    """
    Create numpy_helper initializer for doc_matrix.T.
    If doc_matrix size (bytes) > threshold and AUTO_FALLBACK_TO_FLOAT16 True, convert to float16.
    """
    nbytes = doc_matrix.nbytes
    estimated = nbytes
    print("doc_matrix shape:", doc_matrix.shape, "dtype:", doc_matrix.dtype, "nbytes:", nbytes)
    if estimated > auto_float16_threshold and AUTO_FALLBACK_TO_FLOAT16:
        print(f"WARNING: initializer ~{estimated/1024/1024:.1f}MB exceeds threshold."
              " Converting to float16 to reduce ONNX size.")
        # convert to float16
        doc_matrix = doc_matrix.astype(np.float16)
        # numpy_helper will encode as FLOAT16
        initializer = numpy_helper.from_array(doc_matrix.T, name=name)
        return initializer, doc_matrix
    else:
        initializer = numpy_helper.from_array(doc_matrix.T, name=name)
        return initializer, doc_matrix

def model2onnx(pkl_path: str, onnx_path: str, model_info_path: str, vectorizer_type: str = "tfidf"):
    """
    pkl_path: path to joblib pickle (grid_results_*.pkl)
    vectorizer_type: 'tfidf' or 'count' — used only when rebuilding index from docs
    """
    pkl = Path(pkl_path)
    if not pkl.exists():
        raise FileNotFoundError(f"{pkl_path} not found")

    try:
        searcher = load_searcher(pkl)
    except Exception as e:
        print(f"FATAL: Could not load the full Searcher object. Error: {e}")
        raise

    print_searcher_diag(searcher)

    # If the pickle had a working vectorizer/matrix, prefer to use it.
    if getattr(searcher, "vectorizer", None) is not None and getattr(searcher, "matrix", None) is not None:
        print("Found vectorizer & matrix in pickle. Using them for export.")
        vectorizer = searcher.vectorizer
        mat = searcher.matrix
        # If sparse, convert to dense for ONNX initializer (but watch memory)
        if sp.issparse(mat):
            doc_matrix = mat.toarray().astype(np.float32)
        else:
            doc_matrix = np.asarray(mat).astype(np.float32)
        docs = searcher.docs
    else:
        # Missing vectorizer/matrix (e.g., bm25 not serialized). Rebuild a TFIDF/Count index from docs.
        if not getattr(searcher, "docs", None):
            raise RuntimeError("No docs found in pickle and no vectorizer/matrix available to export.")
        print("\nWARNING: searcher.vectorizer or searcher.matrix missing. Rebuilding vectorizer from raw docs.")
        vec, X, texts = rebuild_vectorizer_from_docs(searcher.docs, vectorizer_type=vectorizer_type)
        vectorizer = vec
        # convert sparse to dense for ONNX; keep as float32 for now
        if sp.issparse(X):
            doc_matrix = X.toarray().astype(np.float32)
        else:
            doc_matrix = np.asarray(X).astype(np.float32)
        # For keys we prefer to keep original doc objects if available
        docs = searcher.docs

    # Validation
    if vectorizer is None:
        raise RuntimeError("Failed to create or retrieve a vectorizer. Aborting ONNX export.")

    # Prepare ONNX graph (MatMul query_vector x doc_matrix.T -> similarity_scores)
    input_name = "query_vector"
    input_size = int(doc_matrix.shape[1])
    X_info = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [None, input_size])

    doc_matrix_name = "doc_matrix"
    # Create initializer safely (auto float16 fallback if huge)
    doc_matrix_initializer, doc_matrix = safe_create_initializer(doc_matrix, doc_matrix_name)

    output_name = "similarity_scores"
    Y_info = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [None, doc_matrix.shape[0]])

    matmul_node = helper.make_node("MatMul", [input_name, doc_matrix_name], [output_name], name="dot_product_node")

    graph = helper.make_graph([matmul_node], "tfidf_search_graph", [X_info], [Y_info], [doc_matrix_initializer])
    onnx_model = helper.make_model(graph, producer_name="vtscada_exporter")
    onnx.save(onnx_model, onnx_path)
    print(f"\n✅ ONNX model saved to {onnx_path}")

    print("num_docs:", doc_matrix.shape[0])
    print("vocab_size:", doc_matrix.shape[1])
    print("doc_matrix dtype:", doc_matrix.dtype)
    print("doc_matrix nbytes:", doc_matrix.nbytes)

    model_info = {
        "vocabulary": getattr(vectorizer, "vocabulary_", {}),
        "keys": [getattr(d, "name", str(i)) for i, d in enumerate(docs)]
    }
    with open(model_info_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    print(f"✅ Model info (vocab, docs) saved to {model_info_path}")

# -------------------------
# Helper: generate docs JSON (unchanged behavior; uses cleaned text)
# -------------------------
def word_wrap(text: str, max_width: int = 70, indent: bool = True) -> List[str]:
    if not text:
        return []
    if max_width <= 0:
        return text.replace(' ', '\n')
    words = text.split()
    if not words:
        return []
    lines = []
    cur_words = []
    cur_len = 0
    for word in words:
        wl = len(word)
        potential = cur_len + (1 if cur_words else 0) + wl
        if potential <= max_width:
            cur_words.append(word)
            cur_len = potential
        else:
            if cur_words:
                lines.append(" ".join(cur_words))
            cur_words = [word]
            cur_len = wl
    if cur_words:
        lines.append(" ".join(cur_words))
    if indent:
        return [lines[0]] + [f"\t{ln}" for ln in lines[1:]] if lines else []
    return lines

def doc_to_snippets(doc) -> List[str]:
    result = []
    fmt = doc.get("format", "")
    if fmt:
        if "(" in fmt:
            idx = fmt.index("(")
            fn_name = clean_text_keep_printable(fmt[:idx].strip())
        else:
            fn_name = clean_text_keep_printable(fmt.strip())
        if doc.get("parameters"):
            result.append(f"{fn_name}(")
            for i, param in enumerate(doc["parameters"]):
                param_desc = clean_text_keep_printable(param.get("description", ""))
                comment = "{ " + param_desc + " }" if param_desc.strip() else ""
                if i < (len(doc["parameters"]) - 1):
                    for line in word_wrap(f"\t{param['name']}, {comment}"):
                        result.append(line)
                else:
                    for line in word_wrap(f"\t{param['name']} {comment}"):
                        result.append(line)
            result.append(");")
        else:
            result.append(f"{fn_name}();")
    return result

def doc_to_comments(doc, max_with_word_wrap=70) -> List[str]:
    description = clean_text_keep_printable((doc.get("description") or "").strip())
    comments = clean_text_keep_printable((doc.get("comments") or "").strip())
    out = []
    if description:
        out += word_wrap("{ Description: " + description + " }", max_width=max_with_word_wrap)
    if comments:
        out += word_wrap("{ " + comments + " }", max_width=max_with_word_wrap)
    return out

# -------------------------
# CLI usage example
# -------------------------
if __name__ == "__main__":
    # Example behavior: export English (existing) and Chinese pickles
    # Adjust paths below as needed.
    model2onnx(
        pkl_path="models/grid_results_en.pkl",
        onnx_path="resources/model.en.onnx",
        model_info_path="resources/info.en.json",
        vectorizer_type="tfidf",
    )

    # For Chinese datasets, we recommend rebuilding using Count or TFIDF with jieba.
    model2onnx(
        pkl_path="models/grid_results_zh-TW.pkl",
        onnx_path="resources/model.zh-tw.onnx",
        model_info_path="resources/info.zh-tw.json",
        vectorizer_type="tfidf",  # or "count" if you prefer counts
    )

    model2onnx(
        pkl_path="models/grid_results_zh-CN.pkl",
        onnx_path="resources/model.zh-cn.onnx",
        model_info_path="resources/info.zh-cn.json",
        vectorizer_type="tfidf",
    )

    # Generate docs resources (unchanged semantics)
    with open("models/vtscada_functions_en.json", "r", encoding="utf-8") as f:
        func_en = json.load(f)
    docs_en = {}
    for doc in func_en:
        docs_en[doc["name"]] = {
            "comments": doc_to_comments(doc),
            "snippets": doc_to_snippets(doc),
            "is_steady_state": "steady" in (doc.get("usage") or "").lower(),
            "is_script": "script" in (doc.get("usage") or "").lower()
        }
    with open("resources/docs.en.json", "w", encoding="utf-8") as f:
        json.dump(docs_en, f, ensure_ascii=False, indent=4)

    # Chinese docs generation: preserve English snippets where possible, but regenerate comments
    for lang_code, input_path, out_path in [
        ("zh-tw", "models/vtscada_functions_zh-TW.json", "resources/docs.zh-tw.json"),
        ("zh-cn", "models/vtscada_functions_zh-CH.json", "resources/docs.zh-cn.json"),
    ]:
        with open(input_path, "r", encoding="utf-8") as f:
            func_list = json.load(f)
        out_docs = {}
        for doc in func_list:
            name = doc["name"]
            out_docs[name] = {"comments": doc_to_comments(doc, 30)}
            # reuse English fields if exist
            if name in docs_en:
                out_docs[name]["snippets"] = docs_en[name]["snippets"]
                out_docs[name]["is_steady_state"] = docs_en[name]["is_steady_state"]
                out_docs[name]["is_script"] = docs_en[name]["is_script"]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_docs, f, ensure_ascii=False, indent=4)

    print("All exports completed.")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VTScada Retrieval   –  Grid‑search *and* Inference modes
-------------------------------------------------------
* Grid mode (default): exhaustive hyper‑parameter sweep
* Inference mode     : run a single config for given query/queries

Examples
--------
# 1. Run grid search & freeze best
python vtscada_search.py --mode grid \
       --docs vtscada_functions.json \
       --queries eval_queries.json \
       --workers -1 \
       --out grid_results.csv

# -> creates grid_results.pkl in same folder

# 2. Ultra-fast inference using frozen model
python vtscada_search.py --mode infer \
       --model grid_results.pkl \
       --docs vtscada_functions.json \
       --query "acknowledge alarm" --topk 5
"""

from __future__ import annotations

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse, json, math, re, csv, sys
from itertools import product
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from nltk.stem import SnowballStemmer
import time
import joblib
from joblib import Parallel, delayed
import multiprocessing
import datetime
from rank_bm25 import BM25Okapi

# ------------------------------ Data Model ------------------------------
@dataclass
class Parameter:
    name: str = ""
    description: str = ""

@dataclass
class FunctionDoc:
    name: str
    description: str = ""
    function_groups: str = ""
    related_to: str = ""
    parameters: List[Parameter] = field(default_factory=list)
    comments: str = ""
    usage: str = ""
    returns: str = ""

    def to_text(self, name_weight: int = 3) -> str:
        parts: List[str] = []
        if self.name:
            parts.extend([self.name] * name_weight)
        for field_text in [
            self.description, self.function_groups, self.related_to,
            self.comments, self.usage, self.returns,
        ]:
            if field_text:
                parts.append(field_text)
        for p in self.parameters:
            if p.description:
                parts.append(p.description)
        combined = re.sub(r"[^\w\s]", " ", " ".join(parts))
        return re.sub(r"\s+", " ", combined).strip().lower()

# ------------------------- Vectorizers (stemming) -----------------------
class StemmedCountVectorizer(CountVectorizer):
    def __init__(self, language: str = "english", **kwargs):
        self.language = language                       # <- store language
        self.stemmer = SnowballStemmer(language)
        kwargs.setdefault("stop_words", language)
        super().__init__(**kwargs)

    def build_analyzer(self):
        base_analyzer = super().build_analyzer()
        stem = self.stemmer.stem
        return lambda doc: (stem(w) for w in base_analyzer(doc))


class StemmedTfidfVectorizer(TfidfVectorizer):
    def __init__(self, language: str = "english", **kwargs):
        self.language = language                       # <- store language
        self.stemmer = SnowballStemmer(language)
        kwargs.setdefault("stop_words", language)
        super().__init__(**kwargs)

    def build_analyzer(self):
        base_analyzer = super().build_analyzer()
        stem = self.stemmer.stem
        return lambda doc: (stem(w) for w in base_analyzer(doc))

# ------------------------------ Searcher --------------------------------
class VTScadaSearcher:
    def __init__(
        self,
        vectorizer_type="tfidf",
        use_stemming=True,
        min_df=1,
        max_df=1.0,
        ngram_range=(1, 1),
        name_weight=3,
        n_clusters: Optional[int] = None,
        kmeans_n_init=10,
        random_state=3,
        dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        dense_normalize: bool = True,
        dense_batch: int = 256,
        dense_device: str = "cpu",
    ):
        self.cfg = dict(
            vectorizer_type=vectorizer_type, use_stemming=use_stemming,
            min_df=min_df, max_df=max_df, ngram_range=ngram_range,
            name_weight=name_weight, n_clusters=n_clusters,
            dense_model=dense_model, dense_normalize=dense_normalize,
            dense_batch=dense_batch, dense_device=dense_device
        )
        self.vectorizer = None
        self.matrix = None
        self.docs: List[FunctionDoc] = []
        self.km: Optional[KMeans] = None
        self.labels_: Optional[np.ndarray] = None
        self.kmeans_n_init = kmeans_n_init
        self.random_state = random_state
        self.bm25: Optional[BM25Okapi] = None
        self.tok_docs: List[List[str]] = []
        self._embedder = None          # SentenceTransformer (lazy)
        self.dense_matrix: Optional[np.ndarray] = None  # (N, D) float32

    # ---- internal helpers ----
    def _make_vectorizer(self):
        common = dict(min_df=self.cfg["min_df"],
                      max_df=self.cfg["max_df"],
                      ngram_range=self.cfg["ngram_range"])
        if self.cfg["vectorizer_type"] == "count":
            if self.cfg["use_stemming"]:
                return StemmedCountVectorizer(**common)
            return CountVectorizer(stop_words="english", **common)
        if self.cfg["vectorizer_type"] == "tfidf":
            if self.cfg["use_stemming"]:
                return StemmedTfidfVectorizer(**common)
            return TfidfVectorizer(stop_words="english", **common)
        if self.cfg["vectorizer_type"] in ("bm25", "dense"):
            return None  # handled separately
        raise ValueError("Unknown vectorizer_type")

    # ---- public API ----
    def index(self, docs: List[FunctionDoc]):
        self.docs = docs
        texts = [d.to_text(self.cfg["name_weight"]) for d in docs]

        if self.cfg["vectorizer_type"] == "bm25":
            # ----- BM25 path -----
            self.tok_docs = [t.split() for t in texts]
            self.bm25 = BM25Okapi(self.tok_docs)
            return

        if self.cfg["vectorizer_type"] == "dense":
            emb = self._get_embedder()
            vecs = emb.encode(
                texts,
                batch_size=self.cfg["dense_batch"],
                convert_to_numpy=True,
                normalize_embeddings=self.cfg["dense_normalize"],
                show_progress_bar=False,
            ).astype("float32")
            self.dense_matrix = vecs
            # optional clustering on dense vectors
            if self.cfg["n_clusters"]:
                self.km = KMeans(
                    n_clusters=self.cfg["n_clusters"],
                    n_init=self.kmeans_n_init,
                    random_state=self.random_state,
                )
                self.labels_ = self.km.fit_predict(self.dense_matrix)
            return

        self.vectorizer = self._make_vectorizer()
        self.matrix = self.vectorizer.fit_transform(texts)
        if self.cfg["n_clusters"]:
            self.km = KMeans(n_clusters=self.cfg["n_clusters"],
                             n_init=self.kmeans_n_init,
                             random_state=self.random_state)
            self.labels_ = self.km.fit_predict(self.matrix)

    def query(self, q: str, top_k=10) -> List[Tuple[int, float]]:
        if self.cfg["vectorizer_type"] == "bm25":
            if self.bm25 is None:
                raise RuntimeError("index() first")
            q_tok = q.lower()
            q_tok = re.sub(r"[^\w\s]", " ", q_tok)
            q_tok = q_tok.split()
            sims = self.bm25.get_scores(q_tok)          # numpy array
            order = sims.argsort()[::-1][:top_k]
            return [(int(i), float(sims[i])) for i in order]

        if self.cfg["vectorizer_type"] == "dense":
            if self.dense_matrix is None:
                raise RuntimeError("index() first")
            emb = self._get_embedder()
            q_clean = " ".join(re.sub(r"[^\w\s]", " ", q.lower()).split())
            q_vec = emb.encode(
                [q_clean],
                convert_to_numpy=True,
                normalize_embeddings=self.cfg["dense_normalize"],
                show_progress_bar=False,
            ).astype("float32")[0]
    
            mask = np.arange(self.dense_matrix.shape[0])
            if self.km is not None and self.labels_ is not None:
                cl = self.km.predict(q_vec.reshape(1, -1))[0]
                mask = np.where(self.labels_ == cl)[0]
                sub = self.dense_matrix[mask]
            else:
                sub = self.dense_matrix
    
            # cosine: if normalized, dot == cosine
            if self.cfg["dense_normalize"]:
                sims = sub @ q_vec
            else:
                # fallback cosine
                sims = (sub @ q_vec) / (np.linalg.norm(sub, axis=1) * np.linalg.norm(q_vec) + 1e-12)
    
            order = sims.argsort()[::-1][:top_k]
            return [(int(mask[i]), float(sims[i])) for i in order]
        
        if self.vectorizer is None:
            raise RuntimeError("index() first")
        q_vec = self.vectorizer.transform(
            [" ".join(re.sub(r"[^\w\s]", " ", q.lower()).split())]
        )
        mask = np.arange(self.matrix.shape[0])
        if self.km is not None and self.labels_ is not None:
            cl = self.km.predict(q_vec)[0]
            mask = np.where(self.labels_ == cl)[0]
        sims = cosine_similarity(self.matrix[mask], q_vec).ravel()
        order = sims.argsort()[::-1][:top_k]
        return [(int(mask[i]), float(sims[i])) for i in order]

    def _get_embedder(self):
        if self._embedder is not None:
            return self._embedder
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "Dense mode requires 'sentence-transformers'. "
                "Install with: pip install sentence-transformers"
            ) from e
        self._embedder = SentenceTransformer(self.cfg["dense_model"],
                                         device=self.cfg["dense_device"])
        return self._embedder

# ------------------------------- Metrics --------------------------------
def precision_at_k(pred: List[int], gold: set[int], k: int) -> float:
    top = pred[:k]
    rel = sum(1 for i in top if i in gold)
    return rel / k


def recall_at_k(pred: List[int], gold: set[int], k: int) -> float:
    if not gold:
        return 0.0
    top = pred[:k]
    rel = sum(1 for i in top if i in gold)
    return rel / len(gold)


def hit_at_k(pred: List[int], gold: set[int], k: int) -> float:
    return 1.0 if any(i in gold for i in pred[:k]) else 0.0


def mrr(pred: List[int], gold: set[int]) -> float:
    for rank, idx in enumerate(pred, 1):
        if idx in gold:
            return 1.0 / rank
    return 0.0


def dcg_at_k(rel: List[int], k: int) -> float:
    rel = rel[:k]
    return sum((r / math.log2(i + 2)) for i, r in enumerate(rel))


def ndcg_at_k(pred: List[int], gold: set[int], k: int) -> float:
    # binary relevance
    rel = [1 if i in gold else 0 for i in pred[:k]]
    dcg = dcg_at_k(rel, k)
    ideal = dcg_at_k(sorted(rel, reverse=True), k)
    return (dcg / ideal) if ideal > 0 else 0.0

# --------------------------- Utility Loaders ----------------------------
def load_docs(path: Path) -> List[FunctionDoc]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        FunctionDoc(
            name=it.get("name", ""),
            description=it.get("description", ""),
            function_groups=it.get("function_groups", ""),
            related_to=it.get("related_to", ""),
            parameters=[Parameter(**p) for p in it.get("parameters", [])],
            comments=it.get("comments", ""), usage=it.get("usage", ""),
            returns=it.get("returns", "")
        )
        for it in data
    ]

def load_queries(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))

def evaluate(
    searcher: VTScadaSearcher,
    queries: List[Dict],
    name2idx: Dict[str, int],
    k_eval: Tuple[int, ...] = (1, 3, 5, 10),
) -> Dict[str, float]:
    """
    Returns averaged metrics over all queries for the first k_eval[-1] results.
    """

    stats = {f"hit@{k}": [] for k in k_eval}
    stats.update({f"p@{k}": [] for k in k_eval})
    stats.update({f"r@{k}": [] for k in k_eval})
    stats.update({f"ndcg@{k}": [] for k in k_eval})
    mrrs = []

    max_k = max(k_eval)
    t0 = time.perf_counter()

    for ex in queries:
        q = ex["query"]
        
        if "answer" in ex:
            gold_names = {ex["answer"].lower()}
        else:
            gold_names = {g.lower() for g in ex.get("answers", [])}
        
        gold_idx = {name2idx[g] for g in gold_names if g in name2idx}
        

        results = searcher.query(q, top_k=max_k)
        pred_idx = [i for (i, _) in results]

        mrrs.append(mrr(pred_idx, gold_idx))

        for k in k_eval:
            stats[f"hit@{k}"].append(hit_at_k(pred_idx, gold_idx, k))
            stats[f"p@{k}"].append(precision_at_k(pred_idx, gold_idx, k))
            stats[f"r@{k}"].append(recall_at_k(pred_idx, gold_idx, k))
            stats[f"ndcg@{k}"].append(ndcg_at_k(pred_idx, gold_idx, k))

    tot_time = time.perf_counter() - t0
    avg_time = tot_time / len(queries)

    out = {m: float(np.mean(v)) for m, v in stats.items()}
    out.update({"mrr": float(np.mean(mrrs)),
                "tot_inf_s": tot_time,
                "avg_inf_s": avg_time})
    return out

# -----------------------------------------------------------------------
#                           GRID‑SEARCH SECTION
# -----------------------------------------------------------------------
def run_gridsearch(
    docs_path: Path,
    queries_path: Path,
    out_csv: Path,
    k_eval: Tuple[int, ...] = (1, 3, 5, 10),
    show_top: int = 20,
    workers: int = 1,
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    dense_norm: bool = True,
    dense_batch: int = 256,
    dense_device: str = "cpu",
):
    docs = load_docs(docs_path)
    queries = load_queries(queries_path)
    name2idx = {d.name.lower(): i for i, d in enumerate(docs)}

    # param ranges (edit as needed)
    vectorizer_types = ["count", "tfidf", "bm25", "dense"]
    use_stemmings = [True, False]
    min_dfs = [1, 3, 5, 10]
    max_dfs = [0.5, 0.8, 1.0]
    ngram_ranges = [(1, 1), (1, 2)]
    name_weights = [1, 3, 5]
    clusters_list = [None, 20, 50, 100]

    # --- build full cartesian product ---
    full_grid = product(
        vectorizer_types,
        use_stemmings,     
        min_dfs,           
        max_dfs,           
        ngram_ranges,      
        name_weights,      
        clusters_list,     
    )
    
    # --- filter so BM25 varies only by name_weight ---
    def valid(cfg):
        vtype, stem, min_df, max_df, ngr, w_name, n_clusters = cfg
        if vtype == "bm25":
            return (stem and min_df == 1 and max_df == 1.0 and ngr == (1, 1) and n_clusters is None)
        if vtype == "dense":
            return (stem and min_df == 1 and max_df == 1.0 and ngr == (1, 1) and n_clusters is None)
        return True
    
    combos = [c for c in full_grid if valid(c)]
    dense_combos = [c for c in combos if c[0] == "dense"]
    sparse_combos = [c for c in combos if c[0] != "dense"]
    
    total  = len(combos)
    print(f"Total configs to try: {total}")

    # ---------- evaluation wrapper ----------
    def eval_one(cfg):
        (vtype, stem, min_df, max_df, ngr, w_name, n_clusters) = cfg
        searcher = VTScadaSearcher(
            vectorizer_type=vtype, use_stemming=stem,
            min_df=min_df, max_df=max_df,
            ngram_range=ngr, name_weight=w_name,
            n_clusters=n_clusters,
            dense_model=dense_model,
            dense_normalize=dense_norm,
            dense_batch=dense_batch, dense_device=dense_device,
        )
        searcher.index(docs)
        metrics = evaluate(searcher, queries, name2idx, k_eval=k_eval)
        return {
            "vectorizer_type": vtype,
            "use_stemming": stem,
            "min_df": min_df,
            "max_df": max_df,
            "ngram_range": str(ngr),
            "name_weight": w_name,
            "clusters": n_clusters,
            **metrics,
        }
    
    results = []
    
    t0 = time.perf_counter()
    # parallel for sparse
    if sparse_combos:
        results += Parallel(n_jobs=workers, verbose=5)(
            delayed(eval_one)(cfg) for cfg in sparse_combos
        )
    # sequential for dense
    for cfg in dense_combos:
        results.append(eval_one(cfg))
    
    elapsed = time.perf_counter() - t0
    print(f"\nGrid finished in {elapsed/60:.1f} min "
          f"(avg {elapsed/total:.2f}s per config)")

    # sort by primary metric (MRR)
    results = sorted(results, key=lambda r: r["mrr"], reverse=True)

    # write CSV
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved all {len(results)} configs to {out_csv}\n")
    print("Top configs by MRR:")
    for r in results[:show_top]:
        print(
            f"MRR={r['mrr']:.4f}  "
            f"P@5={r['p@5']:.4f}  "
            f"vec={r['vectorizer_type']} stem={r['use_stemming']} "
            f"min_df={r['min_df']} max_df={r['max_df']} "
            f"ngr={r['ngram_range']} name_w={r['name_weight']} "
            f"clusters={r['clusters']}"
        )
    
    best = results[0]
    print("\n🏆 Freezing top config …")

    # Re-index with best hyper-parameters
    searcher_best = VTScadaSearcher(
        vectorizer_type=best["vectorizer_type"],
        use_stemming=best["use_stemming"],
        min_df=best["min_df"],
        max_df=best["max_df"],
        ngram_range=eval(best["ngram_range"]),
        name_weight=best["name_weight"],
        n_clusters=best["clusters"],
    )
    searcher_best.index(docs)

    model_path = out_csv.with_suffix(".pkl")   # eg grid_results.pkl
    save_model(searcher_best, model_path)

# -----------------------------------------------------------------------

# ------------------------- INFERENCE FUNCTION --------------------------
def run_inference(
    docs_path: Path,
    query: Optional[str],
    query_file: Optional[Path],
    top_k: int,
    vec_type: str,
    stem: bool,
    min_df: int,
    max_df: float,
    ngram: Tuple[int, int],
    name_weight: int,
    clusters: Optional[int],
    model: Optional[Path] = None,
    dense_model: str="sentence-transformers/all-MiniLM-L6-v2",
    dense_norm: bool=True,
    dense_batch: int=256,
    dense_device: str="cpu",
):
    print("docs_path", docs_path)
    print("query", query)
    print("query_file", query_file)
    print("top_k", top_k)
    print("vec_type", vec_type)
    print("stem", stem)
    print("min_df", min_df)
    print("max_df", max_df)
    print("ngram", ngram)
    print("name_weight", name_weight)
    print("clusters", clusters)
    print("model", model)
    print("dense_model", dense_model)
    print("dense_norm", dense_norm)
    print("dense_batch", dense_batch)
    print("dense_device", dense_device)
    
    if (query is None) == (query_file is None):
        sys.exit("Provide either --query or --query-file (exactly one).")

    if model:
        # -------- fast path: load frozen model ----------
        searcher = load_model(model)
        print(f"model loaded", model)
        docs = searcher.docs
    else:
        docs = load_docs(docs_path)
        searcher = VTScadaSearcher(
            vectorizer_type=vec_type,
            use_stemming=stem,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram,
            name_weight=name_weight,
            n_clusters=clusters,
            dense_model=dense_model,
            dense_normalize=dense_norm,
            dense_batch=dense_batch, dense_device=dense_device,
        )
        searcher.index(docs)

    queries = []
    if query:
        queries = [query]
    elif query_file:
        raw = load_queries(query_file)
        queries = [item["query"] for item in raw]
    else:
        return []

    t0 = time.perf_counter()

    result = []
    for q in queries:
        print(f"\nQuery: {q!r}")
        print("-" * 60)
        for rank, (idx, sim) in enumerate(searcher.query(q, top_k=top_k), 1):
            doc = searcher.docs[idx]
            result.append(doc)
            print(f"{rank:2d}. {doc.name:25s}  cos={sim:.4f}")
            if doc.description:
                print("     ", doc.description[:110] + ("..." if len(doc.description) > 110 else ""))
        print("-" * 60)

    total_time = time.perf_counter() - t0
    avg_time   = total_time / len(queries)
    print(f"\n↳ Inference finished on {len(queries)} query(s)")
    print(f"   • Total time   : {total_time:.4f} s")
    print(f"   • Avg per query: {avg_time:.4f} s")
    return result

# -------------------------------------------------------------
def save_model(searcher: VTScadaSearcher, out_file: Path) -> None:
    """
    Persist everything needed for inference: vectoriser, CSR matrix, docs, labels.
    """
    payload = dict(
        cfg=searcher.cfg,
        vectorizer=searcher.vectorizer,
        matrix=searcher.matrix,
        docs=searcher.docs,
        labels=searcher.labels_,
        bm25=searcher.bm25,        
        tok_docs=searcher.tok_docs,
        dense_matrix=searcher.dense_matrix,
    )
    joblib.dump(payload, out_file)
    print(f"✔ Model saved to {out_file}  ({out_file.stat().st_size/1_048_576:.2f} MB)")

def load_model(pkl: Path) -> VTScadaSearcher:
    """
    Recreate a searcher in inference-ready state.
    """
    payload = joblib.load(pkl)
    s = VTScadaSearcher(**{k: payload["cfg"][k] for k in payload["cfg"]})
    s.vectorizer = payload["vectorizer"]
    s.matrix     = payload["matrix"]
    s.docs       = payload["docs"]
    s.labels_    = payload["labels"]
    s.bm25 = payload.get("bm25")
    s.tok_docs = payload.get("tok_docs", [])
    s.dense_matrix = payload.get("dense_matrix")
    return s

# -----------------------------------------------------------------------
#                                 CLI
# -----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("VTScada grid search & inference")
    p.add_argument("--mode", choices=["grid", "infer"], default="grid")

    # common
    p.add_argument("--docs", required=True, help="vtscada_functions.json")

    # grid‑specific
    p.add_argument("--queries", help="(grid) labeled query file")
    p.add_argument("--out", default="grid_results.csv")
    p.add_argument("--k-eval", default="1,3,5,10")
    p.add_argument("--show-top", type=int, default=20)
    p.add_argument("--workers", type=int, default=1,
               help="Parallel workers for grid mode (default 1 = sequential; "
                    "use -1 for all cores)")
    p.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2",
               help="Dense model name or local path (used in grid/infer when vec=dense)")
    p.add_argument("--dense-batch", type=int, default=256,
                   help="Batch size for dense encoding (grid/infer)")
    p.add_argument("--dense-normalize", dest="dense_norm", action="store_true", default=True,
                   help="L2-normalize dense embeddings (default)")
    p.add_argument("--no-dense-normalize", dest="dense_norm", action="store_false",
                   help="Disable L2-normalization for dense embeddings")
    p.add_argument("--dense-device", default="cpu", choices=["cpu", "cuda"],
               help="Device for dense embeddings (default cpu)")

    # inference‑specific
    g = p.add_argument_group("inference")
    g.add_argument("--query", help="single query string")
    g.add_argument("--query-file", help="JSON array of {'query': ...}")
    g.add_argument("--topk", type=int, default=5)

    # vectoriser knobs for inference
    g.add_argument("--vec", choices=["count", "tfidf", "bm25", "dense"], default="tfidf")
    g.add_argument("--stem", dest="stem", default=True)
    g.add_argument("--min-df", type=int, default=1)
    g.add_argument("--max-df", type=float, default=1.0)
    g.add_argument("--ngram", default="1,1", help="e.g. 1,2 for bigrams")
    g.add_argument("--name-weight", type=int, default=3)
    g.add_argument("--clusters", type=int, default=None)
    g.add_argument("--model", help="(infer) path to .pkl produced by grid mode")
    return p.parse_args()

def main():
    args = parse_args()
    if getattr(args, "dense_device", "cpu") == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    if args.mode == "grid":
        if not args.queries:
            sys.exit("Grid mode requires --queries")
        from math import sqrt  # keep eval functions as‑is
        k_eval = tuple(int(x) for x in args.k_eval.split(","))
        # call previous run_gridsearch (not shown for brevity)
        run_gridsearch(
            docs_path=Path(args.docs),
            queries_path=Path(args.queries),
            out_csv=Path(args.out),
            k_eval=k_eval,
            show_top=args.show_top,
            workers=args.workers,
            dense_model=args.dense_model,
            dense_norm=args.dense_norm,
            dense_batch=args.dense_batch,
            dense_device=args.dense_device,
        )
    else:  # inference
        ngram = tuple(int(x) for x in args.ngram.split(","))
        run_inference(
            docs_path=Path(args.docs),
            query=args.query,
            query_file=Path(args.query_file) if args.query_file else None,
            top_k=args.topk,
            vec_type=args.vec,
            stem=args.stem,
            min_df=args.min_df,
            max_df=args.max_df,
            ngram=ngram,
            name_weight=args.name_weight,
            clusters=args.clusters,
            model=Path(args.model) if args.model else None,
            dense_model=args.dense_model,
            dense_norm=args.dense_norm,
            dense_batch=args.dense_batch,
            dense_device=args.dense_device,
        )

if __name__ == "__main__":
    main()
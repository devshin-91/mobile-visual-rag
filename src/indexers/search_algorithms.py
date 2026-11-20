"""
Mobile-Oriented Vector & Hybrid Search Algorithms Benchmark Suite:
1. S0_flat_fp32       - Exact Brute-Force Inner Product (FP32)
2. S1_sq8_int8        - Symmetric Scalar Quantization (INT8, 4x RAM reduction)
3. S2_bq_hamming      - Binary Quantization (1-bit packbits + POPCNT Hamming, 32x RAM reduction)
4. S3_bq_rescore      - Two-Stage Mobile Search (BQ Stage-1 oversample -> FP32 Stage-2 rescore)
5. S4_hnsw            - HNSW Graph ANN (hnswlib, M=16, ef=48)
6. S5_ivfpq           - FAISS IVF-PQ (Clustered Product Quantization)
7. S6_hybrid_bm25_rrf - Hybrid Dense + BM25 Lexical Fusion via Reciprocal Rank Fusion (k=60)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import faiss
import hnswlib
import numpy as np
from rank_bm25 import BM25Okapi

# Precomputed 8-bit popcount lookup table for ultra-fast vectorized Hamming distance
_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


@dataclass
class SearchOutput:
    indices: list[int]
    scores: list[float]
    latency_ms: float
    index_ram_bytes: int


class BaseVisualIndex:
    name: str = "base"

    def build(self, vectors: np.ndarray, texts: list[str] | None = None) -> None:
        raise NotImplementedError

    def search(self, query_vec: np.ndarray, query_text: str = "", top_k: int = 10) -> SearchOutput:
        raise NotImplementedError

    def memory_bytes(self) -> int:
        raise NotImplementedError


class FlatFP32Index(BaseVisualIndex):
    name = "S0_flat_fp32"

    def build(self, vectors: np.ndarray, texts: list[str] | None = None) -> None:
        self.vecs = np.ascontiguousarray(_l2_normalize(vectors.astype(np.float32)))

    def memory_bytes(self) -> int:
        return int(self.vecs.nbytes)

    def search(self, query_vec: np.ndarray, query_text: str = "", top_k: int = 10) -> SearchOutput:
        q = _l2_normalize(query_vec.astype(np.float32).reshape(1, -1))[0]
        t0 = time.perf_counter()
        sims = self.vecs @ q
        k = min(top_k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k] if k < len(sims) else np.arange(len(sims))
        idx = idx[np.argsort(-sims[idx])]
        ms = (time.perf_counter() - t0) * 1000.0
        return SearchOutput(idx.tolist(), sims[idx].tolist(), ms, self.memory_bytes())


class ScalarQuantizedINT8Index(BaseVisualIndex):
    name = "S1_sq8_int8"

    def build(self, vectors: np.ndarray, texts: list[str] | None = None) -> None:
        normed = _l2_normalize(vectors.astype(np.float32))
        self.scales = np.maximum(np.max(np.abs(normed), axis=1, keepdims=True) / 127.0, 1e-8).astype(np.float16)
        self.qvecs = np.clip(np.round(normed / self.scales.astype(np.float32)), -127, 127).astype(np.int8)

    def memory_bytes(self) -> int:
        return int(self.qvecs.nbytes + self.scales.nbytes)

    def search(self, query_vec: np.ndarray, query_text: str = "", top_k: int = 10) -> SearchOutput:
        q = _l2_normalize(query_vec.astype(np.float32).reshape(1, -1))[0]
        q_scale = max(float(np.max(np.abs(q))) / 127.0, 1e-8)
        q_int8 = np.clip(np.round(q / q_scale), -127, 127).astype(np.int8)
        t0 = time.perf_counter()
        # Int32 accumulation avoids overflow, then rescales per vector
        raw_dots = self.qvecs.astype(np.int32) @ q_int8.astype(np.int32)
        sims = raw_dots.astype(np.float32) * (self.scales[:, 0].astype(np.float32) * q_scale)
        k = min(top_k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k] if k < len(sims) else np.arange(len(sims))
        idx = idx[np.argsort(-sims[idx])]
        ms = (time.perf_counter() - t0) * 1000.0
        return SearchOutput(idx.tolist(), sims[idx].tolist(), ms, self.memory_bytes())


class BinaryQuantizedIndex(BaseVisualIndex):
    name = "S2_bq_hamming"

    def build(self, vectors: np.ndarray, texts: list[str] | None = None) -> None:
        normed = _l2_normalize(vectors.astype(np.float32))
        self.dim = normed.shape[1]
        self.bits = np.packbits((normed >= 0).astype(np.uint8), axis=1)

    def memory_bytes(self) -> int:
        return int(self.bits.nbytes)

    def search(self, query_vec: np.ndarray, query_text: str = "", top_k: int = 10) -> SearchOutput:
        q = _l2_normalize(query_vec.astype(np.float32).reshape(1, -1))
        q_bits = np.packbits((q >= 0).astype(np.uint8), axis=1)[0]
        t0 = time.perf_counter()
        xor_bytes = np.bitwise_xor(self.bits, q_bits)
        hamming = _POPCOUNT_LUT[xor_bytes].sum(axis=1)
        sims = 1.0 - (2.0 * hamming.astype(np.float32) / float(self.dim))
        k = min(top_k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k] if k < len(sims) else np.arange(len(sims))
        idx = idx[np.argsort(-sims[idx])]
        ms = (time.perf_counter() - t0) * 1000.0
        return SearchOutput(idx.tolist(), sims[idx].tolist(), ms, self.memory_bytes())


class BinaryQuantizedRescoreIndex(BaseVisualIndex):
    """
    Mobile Two-Stage Retrieval:
    - Keeps compact 1-bit codes in hot RAM for fast Hamming candidate generation.
    - Fetches exact vectors only for top (oversample_factor * top_k) candidates to rescore.
    """
    name = "S3_bq_rescore"

    def __init__(self, oversample_factor: int = 5) -> None:
        self.oversample_factor = oversample_factor

    def build(self, vectors: np.ndarray, texts: list[str] | None = None) -> None:
        self.vecs = np.ascontiguousarray(_l2_normalize(vectors.astype(np.float32)))
        self.dim = self.vecs.shape[1]
        self.bits = np.packbits((self.vecs >= 0).astype(np.uint8), axis=1)

    def memory_bytes(self) -> int:
        # Active hot search RAM = binary index (FP32 vectors can stay mmap'd on flash)
        return int(self.bits.nbytes)

    def search(self, query_vec: np.ndarray, query_text: str = "", top_k: int = 10) -> SearchOutput:
        q = _l2_normalize(query_vec.astype(np.float32).reshape(1, -1))[0]
        q_bits = np.packbits((q >= 0).astype(np.uint8))
        t0 = time.perf_counter()
        xor_bytes = np.bitwise_xor(self.bits, q_bits)
        hamming = _POPCOUNT_LUT[xor_bytes].sum(axis=1)
        cand_k = min(len(hamming), max(top_k * self.oversample_factor, 20))
        cands = np.argpartition(hamming, cand_k - 1)[:cand_k] if cand_k < len(hamming) else np.arange(len(hamming))
        rescores = self.vecs[cands] @ q
        order = np.argsort(-rescores)[: min(top_k, len(cands))]
        idx = cands[order]
        scores = rescores[order]
        ms = (time.perf_counter() - t0) * 1000.0
        return SearchOutput(idx.tolist(), scores.tolist(), ms, self.memory_bytes())


class HNSWIndex(BaseVisualIndex):
    name = "S4_hnsw"

    def __init__(self, M: int = 16, ef_construction: int = 100, ef_search: int = 48) -> None:
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search

    def build(self, vectors: np.ndarray, texts: list[str] | None = None) -> None:
        normed = _l2_normalize(vectors.astype(np.float32))
        self.n, self.dim = normed.shape
        self.index = hnswlib.Index(space="ip", dim=self.dim)
        self.index.init_index(max_elements=self.n, ef_construction=self.ef_construction, M=self.M)
        self.index.add_items(normed, np.arange(self.n))
        self.index.set_ef(max(self.ef_search, 32))

    def memory_bytes(self) -> int:
        # Vectors + multi-layer adjacency graph overhead (~2 * M * 4 bytes/node + level0 2*M*4)
        vec_bytes = self.n * self.dim * 4
        graph_bytes = self.n * (self.M * 2 * 4 + self.M * 4)
        return int(vec_bytes + graph_bytes)

    def search(self, query_vec: np.ndarray, query_text: str = "", top_k: int = 10) -> SearchOutput:
        q = _l2_normalize(query_vec.astype(np.float32).reshape(1, -1))
        k = min(top_k, self.n)
        self.index.set_ef(max(self.ef_search, k + 10))
        t0 = time.perf_counter()
        labels, dists = self.index.knn_query(q, k=k)
        ms = (time.perf_counter() - t0) * 1000.0
        sims = (1.0 - dists[0]).tolist()
        return SearchOutput(labels[0].tolist(), sims, ms, self.memory_bytes())


class IVFPQIndex(BaseVisualIndex):
    name = "S5_ivfpq"

    def build(self, vectors: np.ndarray, texts: list[str] | None = None) -> None:
        normed = np.ascontiguousarray(_l2_normalize(vectors.astype(np.float32)))
        self.n, self.dim = normed.shape
        # Pick valid subquantizer count M dividing dim and cluster count suitable for N
        m = 32 if self.dim % 32 == 0 else (16 if self.dim % 16 == 0 else 8)
        if self.n >= 512:
            nlist = min(64, int(math.sqrt(self.n)))
            nbits = 8
        else:
            nlist = max(1, min(4, self.n // 32))
            nbits = 4
        quantizer = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIVFPQ(quantizer, self.dim, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
        self.index.train(normed)
        self.index.add(normed)
        self.index.nprobe = min(nlist, max(4, nlist // 4))
        self.m = m
        self.nbits = nbits
        self.nlist = nlist

    def memory_bytes(self) -> int:
        codes_bytes = int(math.ceil(self.m * self.nbits / 8.0) * self.n)
        centroids_bytes = self.nlist * self.dim * 4 + (2**self.nbits) * self.dim * 4
        return int(codes_bytes + centroids_bytes)

    def search(self, query_vec: np.ndarray, query_text: str = "", top_k: int = 10) -> SearchOutput:
        q = np.ascontiguousarray(_l2_normalize(query_vec.astype(np.float32).reshape(1, -1)))
        k = min(top_k, self.n)
        t0 = time.perf_counter()
        dists, labels = self.index.search(q, k)
        ms = (time.perf_counter() - t0) * 1000.0
        valid = [(int(l), float(d)) for l, d in zip(labels[0], dists[0]) if l >= 0]
        return SearchOutput([v[0] for v in valid], [v[1] for v in valid], ms, self.memory_bytes())


class HybridBM25RRFIndex(BaseVisualIndex):
    """Reciprocal Rank Fusion (Cormack et al., 2009) of Exact Dense + BM25 Lexical."""
    name = "S6_hybrid_bm25_rrf"

    def __init__(self, rrf_k: int = 60, dense_weight: float = 1.0, bm25_weight: float = 0.7) -> None:
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    def build(self, vectors: np.ndarray, texts: list[str] | None = None) -> None:
        self.vecs = np.ascontiguousarray(_l2_normalize(vectors.astype(np.float32)))
        docs = texts or [""] * len(self.vecs)
        self.tokenized = [d.lower().split() if d.strip() else ["empty"] for d in docs]
        self.bm25 = BM25Okapi(self.tokenized)
        self._text_bytes = sum(len(d.encode("utf-8")) for d in docs)

    def memory_bytes(self) -> int:
        return int(self.vecs.nbytes + self._text_bytes)

    def search(self, query_vec: np.ndarray, query_text: str = "", top_k: int = 10) -> SearchOutput:
        q = _l2_normalize(query_vec.astype(np.float32).reshape(1, -1))[0]
        t0 = time.perf_counter()
        dense_sims = self.vecs @ q
        dense_order = np.argsort(-dense_sims)
        bm25_scores = self.bm25.get_scores(query_text.lower().split() if query_text.strip() else ["empty"])
        bm25_order = np.argsort(-bm25_scores)

        rrf_scores = np.zeros(len(self.vecs), dtype=np.float32)
        for rank, idx in enumerate(dense_order):
            rrf_scores[idx] += self.dense_weight / (self.rrf_k + rank + 1)
        if np.max(bm25_scores) > 0:
            for rank, idx in enumerate(bm25_order):
                if bm25_scores[idx] > 0:
                    rrf_scores[idx] += self.bm25_weight / (self.rrf_k + rank + 1)

        k = min(top_k, len(rrf_scores))
        top_idx = np.argsort(-rrf_scores)[:k]
        ms = (time.perf_counter() - t0) * 1000.0
        return SearchOutput(top_idx.tolist(), rrf_scores[top_idx].tolist(), ms, self.memory_bytes())


def build_index(algo_name: str, vectors: np.ndarray, texts: list[str] | None = None) -> BaseVisualIndex:
    registry: dict[str, type[BaseVisualIndex]] = {
        "S0_flat_fp32": FlatFP32Index,
        "S1_sq8_int8": ScalarQuantizedINT8Index,
        "S2_bq_hamming": BinaryQuantizedIndex,
        "S3_bq_rescore": BinaryQuantizedRescoreIndex,
        "S4_hnsw": HNSWIndex,
        "S5_ivfpq": IVFPQIndex,
        "S6_hybrid_bm25_rrf": HybridBM25RRFIndex,
    }
    if algo_name not in registry:
        raise ValueError(f"Unknown search algorithm: {algo_name}")
    idx = registry[algo_name]()
    idx.build(vectors, texts=texts)
    return idx


def benchmark_10k_mobile_scaling(dim: int = 512, n_items: int = 10_000, n_queries: int = 30) -> list[dict[str, Any]]:
    """
    Measure exact Recall@10 vs Ground-Truth Flat FP32, RAM (MB), and P50/P95 search latency (ms)
    at realistic smartphone photo gallery scale (10,000 indexed visual items).
    """
    rng = np.random.default_rng(42)
    base_clusters = _l2_normalize(rng.normal(size=(50, dim)).astype(np.float32))
    assign = rng.integers(0, 50, size=n_items)
    vecs = _l2_normalize(base_clusters[assign] + 0.35 * rng.normal(size=(n_items, dim)).astype(np.float32))
    queries = _l2_normalize(base_clusters[rng.integers(0, 50, size=n_queries)] + 0.30 * rng.normal(size=(n_queries, dim)).astype(np.float32))
    dummy_texts = [f"photo scene cluster_{c} item_{i}" for i, c in enumerate(assign)]

    flat = build_index("S0_flat_fp32", vecs, dummy_texts)
    gt_top10 = [set(flat.search(q, top_k=10).indices) for q in queries]

    rows = []
    for algo in ["S0_flat_fp32", "S1_sq8_int8", "S2_bq_hamming", "S3_bq_rescore", "S4_hnsw", "S5_ivfpq", "S6_hybrid_bm25_rrf"]:
        idx = build_index(algo, vecs, dummy_texts)
        latencies = []
        recalls = []
        for q, gt in zip(queries, gt_top10):
            res = idx.search(q, query_text="photo scene cluster_0", top_k=10)
            latencies.append(res.latency_ms)
            recalls.append(len(set(res.indices) & gt) / 10.0)
        rows.append({
            "algorithm": algo,
            "n_items": n_items,
            "dim": dim,
            "ram_mb": round(idx.memory_bytes() / (1024**2), 3),
            "recall_at_10_vs_exact": round(float(np.mean(recalls)), 4),
            "latency_p50_ms": round(float(np.percentile(latencies, 50)), 3),
            "latency_p95_ms": round(float(np.percentile(latencies, 95)), 3),
        })
    return rows

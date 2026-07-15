"""Core narrative-extraction pipeline for the demo.

The pipeline is:

    query --> LLM decomposition --> top-k retrieval --> endpoint selection
          --> narrative extraction (max-capacity path) --> summary

plus the UMAP + HDBSCAN landscape projection used for visualization.

Everything is embeddings-first: the pre-computed document embeddings for each
dataset are loaded from disk, so no local embedding model is required. Only the
query embedding, endpoint embeddings, query decomposition and summary need
live OpenAI API calls.
"""

import heapq
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from openai import OpenAI
from pydantic import BaseModel

from demo_datasets import DatasetConfig

# --------------------------------------------------------------------------- #
# Model configuration
# --------------------------------------------------------------------------- #
EMBEDDING_MODEL = "text-embedding-3-small"
QUERY_EXPANSION_MODEL = "gpt-5.4"
NARRATIVE_SUMMARY_MODEL = "gpt-5.4"

TOP_K = 1000               # size of the retrieved candidate set
ENDPOINT_QUERY_WEIGHT = 0.75   # blend of endpoint embedding vs. raw query
MIN_YEARS_APART = 2        # temporal endpoint separation (VisPub)


# --------------------------------------------------------------------------- #
# OpenAI helpers
# --------------------------------------------------------------------------- #
def _client() -> OpenAI:
    return OpenAI()


def get_openai_embeddings(texts: list[str], model: str = EMBEDDING_MODEL) -> torch.Tensor:
    texts = [t.replace("\n", " ") for t in texts]
    response = _client().embeddings.create(input=texts, model=model)
    return torch.tensor([d.embedding for d in response.data])


class NarrativeEndpointQueries(BaseModel):
    source_query: str
    target_query: str


def expand_query(query: str, system_prompt: str) -> tuple[str, str]:
    """Decompose a user query into a source and a target search query."""
    response = _client().chat.completions.parse(
        model=QUERY_EXPANSION_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        response_format=NarrativeEndpointQueries,
        temperature=0,
    )
    parsed = response.choices[0].message.parsed
    source = parsed.source_query.strip()
    target = parsed.target_query.strip()
    if not source or not target:
        raise ValueError("Query decomposition returned empty sub-queries.")
    return source, target


def summarize_narrative(query: str, semantic_data: pd.DataFrame, path: list[int],
                        cfg: DatasetConfig) -> str:
    """Synthesize a one-paragraph, in-text-cited summary of the narrative."""
    listing = ""
    for i, idx in enumerate(path):
        row = semantic_data.iloc[idx]
        title = row[cfg.title_col]
        date = row[cfg.date_col]
        abstract = row[cfg.abstract_col]
        listing += f"[{i + 1}] {title} — {date}\nExcerpt: {abstract}\n\n"

    system_prompt = (
        "You are a helpful assistant that synthesizes narratives from collections of documents and "
        "provides concise, coherent answers to queries that follow the specified order of events as the "
        "narrative collections. You always answer the query directly with support from the provided narrative."
    )
    prompt = (
        f'The following documents form a narrative sequence extracted in response to the query: "{query}".\n\n'
        f"<START_NARRATIVE>\n{listing}\n<END_NARRATIVE>\n\n"
        "Provide a concise, one-paragraph response addressing the query directly while capturing the main thread or insights "
        "that runs through these documents and connects them to the query. Synthesize the content into a coherent story that "
        "*MUST* follow the same order of events as the narrative provided above. Do not list or enumerate the documents individually. "
        "Instead, reference documents inline using the format `[idx]` where `idx` is the document's position in the list (starting at 1). "
        "If multiple documents support the same point, cite them individually (e.g., [1], [2]) rather than as a range. "
        "All documents must contribute to the response and must be appropriately cited in-text using the previously mentioned format.\n\n"
        "Note that the documents may not follow chronological order, but the narrative flow should be based on the sequence provided in "
        "the list, which is derived from semantic relationships rather than strict temporal progression. Be concise and short."
    )
    response = _client().chat.completions.create(
        model=NARRATIVE_SUMMARY_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


# --------------------------------------------------------------------------- #
# Data / embeddings loading
# --------------------------------------------------------------------------- #
def load_embeddings(embed_folder: Path) -> torch.Tensor:
    """Load the pre-computed OpenAI embeddings for a dataset."""
    path = Path(embed_folder) / "embed_data-gpt4.pickle"
    with open(path, "rb") as handle:
        embeds = pickle.load(handle)
    if not isinstance(embeds, torch.Tensor):
        embeds = torch.tensor(np.asarray(embeds))
    return embeds.float()


def build_citation_network(semantic_data: pd.DataFrame, cfg: DatasetConfig
                           ) -> tuple[torch.Tensor, np.ndarray]:
    """Build the internal citation adjacency and PageRank scores (VisPub).

    ``semantic_data`` is the embeddings-aligned frame, so row i of the returned
    adjacency corresponds to embeddings row i. ``citation_net[i, j] == 1`` means
    document i references document j.
    """
    n = len(semantic_data)
    doi_to_idx = {
        doi: idx
        for idx, doi in enumerate(semantic_data[cfg.doi_col].values)
        if pd.notna(doi)
    }

    citation_net = torch.zeros((n, n), dtype=torch.int8)
    refs = semantic_data[cfg.references_col].values
    for i, ref_field in enumerate(refs):
        if pd.isna(ref_field) or ref_field == "":
            continue
        for ref in str(ref_field).split(";"):
            j = doi_to_idx.get(ref.strip())
            if j is not None:
                citation_net[i, j] = 1

    pagerank_scores = _pagerank(citation_net.numpy(), max_iter=256)
    return citation_net, pagerank_scores


def _pagerank(adj_matrix: np.ndarray, damping: float = 0.85,
              tol: float = 1e-6, max_iter: int = 100) -> np.ndarray:
    A = np.array(adj_matrix, dtype=float)
    n = A.shape[0]
    row_sums = A.sum(axis=1, keepdims=True)
    dangling = (row_sums == 0).flatten()
    row_sums[row_sums == 0] = 1
    A = A / row_sums
    M = A.T
    ranks = np.ones(n) / n
    for _ in range(max_iter):
        dangling_sum = ranks[dangling].sum()
        new_ranks = damping * (M @ ranks + dangling_sum / n) + (1 - damping) / n
        if np.linalg.norm(new_ranks - ranks, 1) < tol:
            return new_ranks
        ranks = new_ranks
    return ranks


# --------------------------------------------------------------------------- #
# Retrieval + endpoint selection
# --------------------------------------------------------------------------- #
def retrieve_top_k(query_embed: torch.Tensor, embeds: torch.Tensor,
                   top_k: int = TOP_K) -> torch.Tensor:
    """Return the indices of the ``top_k`` documents most similar to the query."""
    k = min(top_k, embeds.shape[0])
    _, indices = torch.topk(query_embed @ embeds.T, k=k, dim=1)
    return indices.squeeze(0)


def blend_endpoint_embeddings(src_q: str, tgt_q: str,
                              query_embed: torch.Tensor) -> torch.Tensor:
    endpoint_embeddings = get_openai_embeddings([src_q, tgt_q])
    endpoint_embeddings = (
        ENDPOINT_QUERY_WEIGHT * endpoint_embeddings
        + (1 - ENDPOINT_QUERY_WEIGHT) * query_embed
    )
    return endpoint_embeddings / endpoint_embeddings.norm(dim=1, keepdim=True)


def select_endpoints(endpoint_embeddings: torch.Tensor, embeds: torch.Tensor,
                     top_candidate_indices: torch.Tensor,
                     semantic_data: pd.DataFrame, cfg: DatasetConfig,
                     temporal: bool) -> tuple[int, int]:
    """Choose source/target documents (as *local* top-k positions).

    With ``temporal`` on, pick the highest-scoring pair whose years are at least
    ``MIN_YEARS_APART`` apart (source earlier). Otherwise pick the best-scoring
    document for each endpoint, avoiding a degenerate source == target.
    """
    scores = endpoint_embeddings @ embeds[top_candidate_indices].T
    src_ranked = scores[0].argsort(descending=True).tolist()
    tgt_ranked = scores[1].argsort(descending=True).tolist()

    if not temporal:
        src_local = src_ranked[0]
        tgt_local = next(t for t in tgt_ranked if t != src_local)
        return src_local, tgt_local

    years = semantic_data[cfg.date_col].values
    idx = top_candidate_indices.numpy()
    for src_local in src_ranked:
        src_year = years[idx[src_local]]
        for tgt_local in tgt_ranked:
            if tgt_local == src_local:
                continue
            if years[idx[tgt_local]] - src_year >= MIN_YEARS_APART:
                return src_local, tgt_local
    raise ValueError(
        f"No endpoint pair at least {MIN_YEARS_APART} years apart was found in "
        "the top-k candidates. Try disabling the temporal constraint."
    )


# --------------------------------------------------------------------------- #
# Graph construction + narrative extraction
# --------------------------------------------------------------------------- #
def build_graph(embeds: torch.Tensor, top_candidate_indices: torch.Tensor,
                semantic_data: pd.DataFrame, cfg: DatasetConfig,
                temporal: bool, use_citations: bool, lam: float,
                citation_net: Optional[torch.Tensor] = None,
                pagerank_scores: Optional[np.ndarray] = None) -> torch.Tensor:
    """Assemble the document coherence graph over the top-k candidates."""
    idx = top_candidate_indices
    semantic_graph = embeds[idx] @ embeds[idx].T

    temporal_mask = None
    if temporal:
        years = semantic_data[cfg.date_col].values[idx.numpy()]
        temporal_mask = torch.tensor(years[:, None] <= years[None, :])
        semantic_graph = semantic_graph * temporal_mask

    graph = semantic_graph
    if use_citations and lam > 0 and citation_net is not None:
        # Transpose the citation network so that, under the temporal ordering,
        # the *later* paper in the narrative cites the *earlier* one.
        pr = torch.from_numpy(pagerank_scores).float()
        sub = citation_net.T[idx][:, idx].float() * pr[idx]
        sub = sub / (sub.sum(axis=1, keepdims=True) + 1e-10)
        graph = semantic_graph + lam * sub
        if temporal_mask is not None:
            graph = graph * temporal_mask

    return graph


def _maximum_capacity_path(matrix: np.ndarray, s: int, t: int):
    """Widest-path (max-min capacity) search from ``s`` to ``t``."""
    queue = [(-float("inf"), s)]
    best_min_capacity = {s: float("inf")}
    parent = {s: None}
    while queue:
        neg_cap, node = heapq.heappop(queue)
        min_capacity = -neg_cap
        if node == t:
            path = []
            cur = t
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path
        row = matrix[node]
        for neighbor in np.nonzero(row)[0]:
            capacity = row[neighbor]
            path_min = min(min_capacity, capacity)
            if neighbor not in best_min_capacity or path_min > best_min_capacity[neighbor]:
                best_min_capacity[neighbor] = path_min
                parent[neighbor] = int(node)
                heapq.heappush(queue, (-path_min, int(neighbor)))
    return None


@dataclass
class Narrative:
    path: list[int]            # global document indices (into semantic_data)
    local_path: list[int]      # positions within the top-k candidate set
    coherences: np.ndarray     # edge weights along the path
    bottleneck: float
    reliability: float

    @property
    def weakest_link(self) -> tuple[int, int]:
        wl = int(self.coherences.argmin())
        return self.path[wl], self.path[wl + 1]


def find_narrative(graph: torch.Tensor, src_local: int, tgt_local: int,
                   top_candidate_indices: torch.Tensor) -> Narrative:
    matrix = graph.detach().numpy()
    local_path = _maximum_capacity_path(matrix, src_local, tgt_local)
    if not local_path:
        raise ValueError(
            "No coherent path could be found between the selected endpoints. "
            "Try a different query or relax the constraints."
        )
    idx = top_candidate_indices.numpy()
    path = [int(idx[i]) for i in local_path]
    coherences = np.array([
        matrix[a, b] for a, b in zip(local_path[:-1], local_path[1:])
    ])
    reliability = float(coherences.prod() ** (1 / len(coherences)))
    return Narrative(
        path=path,
        local_path=local_path,
        coherences=coherences,
        bottleneck=float(coherences.min()),
        reliability=reliability,
    )


# --------------------------------------------------------------------------- #
# Landscape projection (UMAP + HDBSCAN)
# --------------------------------------------------------------------------- #
def compute_landscape(embeds: torch.Tensor, semantic_data: pd.DataFrame,
                      cfg: DatasetConfig) -> pd.DataFrame:
    """Project embeddings to 2-D (UMAP) and cluster them (HDBSCAN)."""
    from umap import UMAP
    import hdbscan

    umap_model = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap_embeddings = umap_model.fit_transform(embeds.numpy())

    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=cfg.hdbscan_min_cluster_size,
        min_samples=1,
        prediction_data=True,
    ).fit(umap_embeddings)
    cluster_probs = hdbscan.prediction.all_points_membership_vectors(hdbscan_model)
    clusters = cluster_probs.argmax(1)

    return pd.DataFrame({
        "idx": np.arange(len(umap_embeddings)),
        "x": umap_embeddings[:, 0],
        "y": umap_embeddings[:, 1],
        "cluster": clusters.astype(str),
        "title": semantic_data[cfg.title_col].values,
        "abstract": semantic_data[cfg.abstract_col].values,
        "date": semantic_data[cfg.date_col].astype(str).values,
    })

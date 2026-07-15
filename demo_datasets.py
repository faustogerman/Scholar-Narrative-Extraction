"""Dataset definitions and loaders for the demo.

Two datasets are supported:

* **VisPub** — IEEE Visualization publications. It carries year metadata *and*
  an internal citation network, so it supports the full pipeline: temporal
  constraints + PageRank-weighted citation coherence.

* **Physics** — arXiv High-Energy Physics Theory (hep-th) abstracts. The demo
  uses only the semantic graph for Physics (no temporal constraint, no citation
  information).

Each dataset is loaded from a single ``text_data.csv`` file whose row order is
aligned 1-to-1 with the pre-computed embeddings pickle for that dataset. The
extra columns needed to build the citation network (``DOI`` /
``InternalReferences`` for VisPub) are kept on the same frame so the alignment
is preserved.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

# Data lives in the ``Data/`` folder next to this file. Override the location by
# setting NARRATIVE_DATA_DIR to a folder that contains VisPubData/ and cit-HepTh/.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = (
    Path(os.environ["NARRATIVE_DATA_DIR"]).resolve()
    if os.environ.get("NARRATIVE_DATA_DIR")
    else PROJECT_ROOT / "Data"
)


# --------------------------------------------------------------------------- #
# VisPub
# --------------------------------------------------------------------------- #
def load_vispub() -> pd.DataFrame:
    """Load VisPub, keeping rows with a Title, Abstract and Year.

    The returned frame is aligned with ``data/VisPubData/embed_data-gpt4.pickle``
    (same dropna filter and order used to generate those embeddings). DOI and
    InternalReferences are retained so the citation network can be built on the
    exact same row indices as the embeddings.
    """
    df = pd.read_csv(DATA_DIR / "VisPubData" / "text_data.csv")
    keep = df[["Title", "Abstract", "Year"]].notna().all(axis=1)
    return df[keep].reset_index(drop=True)


VISPUB_EXPANSION_PROMPT = (
    "You are an expert in visualization and human-computer interaction literature narratives, and can "
    "easily decompose a user's query into two semantic search queries: one for the narrative's STARTING "
    "point (source) and one for its ENDING point (target). The narrative flows source -> target. If the "
    "request implies an order or progression (e.g. 'how did X lead to Y', 'from A to B'), set source to "
    "the earlier concept and target to the later one. If no order is implied, choose two distinct, on-topic "
    "facets that make natural endpoints. Each query should be a concise phrase or sentence suitable for "
    "embedding-based retrieval in a corpus of visualization and human-computer interaction research "
    "articles. They should not be questions."
)


# --------------------------------------------------------------------------- #
# Physics (arXiv hep-th)
# --------------------------------------------------------------------------- #
def load_physics() -> pd.DataFrame:
    """Load the hep-th abstracts as an embeddings-aligned semantic frame.

    Read from ``Data/cit-HepTh/text_data.csv``, whose row order matches the
    pre-computed embeddings pickle, so ``semantic_data`` row i lines up with
    embeddings row i.
    """
    df = pd.read_csv(DATA_DIR / "cit-HepTh" / "text_data.csv")
    return df[["title", "abstract", "date"]].dropna().reset_index(drop=True)


PHYSICS_EXPANSION_PROMPT = (
    "You are an expert in high-energy particle physics narratives, and can easily decompose a user's "
    "query into two semantic search queries: one for the narrative's STARTING point (source) and one "
    "for its ENDING point (target). The narrative flows from source to target. If the request implies an "
    "order or progression (e.g. 'how did X lead to Y', 'from A to B'), set source to the earlier concept "
    "and target to the later one. If no order is implied, choose two distinct, on-topic facets that make "
    "natural endpoints. Each query should be a concise phrase or sentence with specific keywords *scoped* to each "
    " endpoint concept and suitable for embedding-based retrieval in a corpus of high-energy physics research "
    "articles. It should not be a question. Do not use the same keywords in both queries; make them distinct "
    "and focused on their respective endpoint concepts. Do not assume extra context or concepts beyond the "
    "ones explicitly mentioned in the user's query."
)


# --------------------------------------------------------------------------- #
# Config registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DatasetConfig:
    key: str
    label: str
    description: str
    loader: Callable[[], pd.DataFrame]
    embed_folder: Path
    title_col: str
    abstract_col: str
    date_col: str
    expansion_prompt: str
    hdbscan_min_cluster_size: int
    supports_temporal: bool
    supports_citations: bool
    # Columns used to build the citation network (VisPub only).
    doi_col: Optional[str] = None
    references_col: Optional[str] = None
    # Example queries shown in the UI.
    example_queries: tuple = field(default_factory=tuple)
    # True when the date column is a plain integer year.
    date_is_year: bool = False


DATASETS = {
    "vispub": DatasetConfig(
        key="vispub",
        label="VisPub — IEEE Visualization papers",
        description=(
            "3,549 IEEE Visualization publications (1990–2022) with abstracts, "
            "publication years, and an internal citation network."
        ),
        loader=load_vispub,
        embed_folder=DATA_DIR / "VisPubData",
        title_col="Title",
        abstract_col="Abstract",
        date_col="Year",
        expansion_prompt=VISPUB_EXPANSION_PROMPT,
        hdbscan_min_cluster_size=32,
        supports_temporal=True,
        supports_citations=True,
        doi_col="DOI",
        references_col="InternalReferences",
        date_is_year=True,
        example_queries=(
            "What is semantic interaction in visualization and how does it relate to user experience?",
            "How did visualization techniques for large graphs evolve over time?",
            "From early information visualization principles to modern immersive analytics.",
        ),
    ),
    "physics": DatasetConfig(
        key="physics",
        label="Physics — arXiv High-Energy Physics Theory (hep-th)",
        description=(
            "29,555 arXiv hep-th abstracts. This demo uses only the semantic "
            "graph for Physics (no temporal or citation constraints)."
        ),
        loader=load_physics,
        embed_folder=DATA_DIR / "cit-HepTh" / "embeddings",
        title_col="title",
        abstract_col="abstract",
        date_col="date",
        expansion_prompt=PHYSICS_EXPANSION_PROMPT,
        hdbscan_min_cluster_size=40,
        supports_temporal=False,
        supports_citations=False,
        example_queries=(
            "What is the chain of reasoning that can lead from string theory to specific predictions about black hole thermodynamics?",
            "How did the AdS/CFT correspondence develop and connect to gauge theories?",
            "From supersymmetry breaking to phenomenological predictions.",
        ),
    ),
}

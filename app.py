import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Environment: load OPENAI_API_KEY from a .env file next to this script.
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent


def _load_dotenv() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

import numpy as np
import pandas as pd
import gradio as gr

import pipeline
import viz
from demo_datasets import DATASETS

DATASET_KEYS = list(DATASETS.keys())
DATASET_CHOICES = [(DATASETS[k].label, k) for k in DATASET_KEYS]


# --------------------------------------------------------------------------- #
# Cached heavy resources (Gradio has no built-in resource cache)
# --------------------------------------------------------------------------- #
_dataset_cache: dict = {}
_landscape_cache: dict = {}
_citation_cache: dict = {}


def load_dataset(key: str):
    if key not in _dataset_cache:
        cfg = DATASETS[key]
        semantic_data = cfg.loader()
        embeds = pipeline.load_embeddings(cfg.embed_folder)
        if len(semantic_data) != embeds.shape[0]:
            raise RuntimeError(
                f"Row mismatch for '{key}': {len(semantic_data)} documents vs. "
                f"{embeds.shape[0]} embeddings. The embeddings pickle is out of sync."
            )
        _dataset_cache[key] = (semantic_data, embeds)
    return _dataset_cache[key]


def load_landscape(key: str):
    if key not in _landscape_cache:
        cfg = DATASETS[key]
        semantic_data, embeds = load_dataset(key)
        _landscape_cache[key] = pipeline.compute_landscape(embeds, semantic_data, cfg)
    return _landscape_cache[key]


def load_citations(key: str):
    if key not in _citation_cache:
        cfg = DATASETS[key]
        semantic_data, _ = load_dataset(key)
        _citation_cache[key] = pipeline.build_citation_network(semantic_data, cfg)
    return _citation_cache[key]


# --------------------------------------------------------------------------- #
# Dataset-dependent UI updates
# --------------------------------------------------------------------------- #
def on_dataset_change(key: str):
    cfg = DATASETS[key]
    return (
        gr.update(value=f"**{cfg.label}**\n\n{cfg.description}"),
        gr.update(
            value=cfg.supports_temporal, interactive=cfg.supports_temporal,
            label="Constrain narrative temporally"
            + ("" if cfg.supports_temporal else "  (unavailable for this dataset)"),
        ),
        gr.update(
            value=False, interactive=cfg.supports_citations,
            label="Include citation PageRank (λ)"
            + ("" if cfg.supports_citations else "  (unavailable for this dataset)"),
        ),
        gr.update(interactive=cfg.supports_citations),
        gr.update(choices=list(cfg.example_queries), value=cfg.example_queries[0]),
        cfg.example_queries[0],
    )


# --------------------------------------------------------------------------- #
# Pipeline execution
# --------------------------------------------------------------------------- #
def run_pipeline(dataset_key, temporal, use_citations, lam, top_k, query,
                 progress=gr.Progress()):
    cfg = DATASETS[dataset_key]
    # Honor per-dataset capabilities regardless of stale widget state.
    temporal = bool(temporal) and cfg.supports_temporal
    use_citations = bool(use_citations) and cfg.supports_citations

    if not query or not query.strip():
        raise gr.Error("Please enter a query.")
    if not os.environ.get("OPENAI_API_KEY"):
        raise gr.Error("Set OPENAI_API_KEY in a .env file (see .env.example) to run the pipeline.")

    semantic_data, embeds = load_dataset(dataset_key)

    progress(0.05, desc="Decomposing the query into narrative endpoints…")
    src_q, tgt_q = pipeline.expand_query(query, cfg.expansion_prompt)

    progress(0.20, desc="Embedding the query and retrieving candidates…")
    query_embed = pipeline.get_openai_embeddings([query])
    top_idx = pipeline.retrieve_top_k(query_embed, embeds, int(top_k))
    endpoint_embeddings = pipeline.blend_endpoint_embeddings(src_q, tgt_q, query_embed)

    progress(0.35, desc="Selecting source and target documents…")
    src_local, tgt_local = pipeline.select_endpoints(
        endpoint_embeddings, embeds, top_idx, semantic_data, cfg, temporal)

    citation_net = pagerank_scores = None
    if use_citations and lam > 0:
        progress(0.45, desc="Building citation network and PageRank scores…")
        citation_net, pagerank_scores = load_citations(dataset_key)

    progress(0.55, desc="Finding the maximum-capacity narrative path…")
    graph = pipeline.build_graph(
        embeds, top_idx, semantic_data, cfg, temporal, use_citations, lam,
        citation_net, pagerank_scores)
    narrative = pipeline.find_narrative(graph, src_local, tgt_local, top_idx)

    progress(0.75, desc="Computing the landscape projection…")
    vis_data = load_landscape(dataset_key)
    chart = viz.narrative_landscape_chart(vis_data, narrative.path, top_idx.numpy())

    progress(0.90, desc="Summarizing the narrative…")
    summary = pipeline.summarize_narrative(query, semantic_data, narrative.path, cfg)

    # ---- Build outputs -----------------------------------------------------
    tags = []
    if temporal:
        tags.append("⏳ temporal")
    if use_citations and lam > 0:
        tags.append(f"🔗 citation λ={lam:.1f}")
    if not tags:
        tags.append("semantic only")

    decomposition_md = (
        "### 1 · Query decomposition\n"
        f"**Source query**\n\n> {src_q}\n\n"
        f"**Target query**\n\n> {tgt_q}\n\n"
        f"*Constraints: {' · '.join(tags)}*"
    )

    src_idx = int(top_idx[src_local].item())
    tgt_idx = int(top_idx[tgt_local].item())
    src_row = semantic_data.iloc[src_idx]
    tgt_row = semantic_data.iloc[tgt_idx]
    endpoints_md = (
        "### 2 · Selected endpoints\n"
        f"**Source** · {src_row[cfg.date_col]}\n\n{src_row[cfg.title_col]}\n\n"
        f"**Target** · {tgt_row[cfg.date_col]}\n\n{tgt_row[cfg.title_col]}"
    )

    metrics_md = (
        "### 4 · Extracted narrative\n"
        f"**Documents:** {len(narrative.path)}  |  "
        f"**Bottleneck coherence:** {narrative.bottleneck:.3f}  |  "
        f"**Reliability:** {narrative.reliability:.3f}"
    )

    coh = list(narrative.coherences) + [np.nan]
    table = pd.DataFrame({
        "#": np.arange(1, len(narrative.path) + 1),
        "idx": narrative.path,
        cfg.date_col: [semantic_data.iloc[i][cfg.date_col] for i in narrative.path],
        "Title": [semantic_data.iloc[i][cfg.title_col] for i in narrative.path],
        "→ coherence": [f"{c:.3f}" if not np.isnan(c) else "" for c in coh],
    })

    summary_md = "### 5 · Narrative summary\n" + summary

    progress(1.0, desc="Done.")
    return decomposition_md, endpoints_md, chart, metrics_md, table, summary_md


def render_landscape(dataset_key, progress=gr.Progress()):
    """Full landscape preview for the currently selected dataset."""
    progress(0.1, desc="Computing the landscape projection…")
    vis_data = load_landscape(dataset_key)
    progress(1.0, desc="Done.")
    return viz.full_landscape_chart(vis_data)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def build_demo() -> gr.Blocks:
    default_key = DATASET_KEYS[0]
    default_cfg = DATASETS[default_key]
    api_ok = bool(os.environ.get("OPENAI_API_KEY"))

    with gr.Blocks(title="Narrative Extraction") as demo:
        gr.Markdown(
            "# 🏔️ Narrative Extraction in Scholarly Research\n"
            "Ask a question over a document collection. The system decomposes it "
            "into narrative endpoints, retrieves candidates, and extracts the most "
            "coherent storyline connecting them — then visualizes and summarizes it."
        )
        gr.Markdown(
            "✅ OpenAI API key detected." if api_ok
            else "⚠️ **No OPENAI_API_KEY found** — set it in a `.env` file (see `.env.example`) to run the pipeline."
        )

        with gr.Row():
            # ---- Configuration column ------------------------------------- #
            with gr.Column(scale=1):
                gr.Markdown("### Configuration")
                dataset_dd = gr.Dropdown(
                    choices=DATASET_CHOICES, value=default_key, label="Dataset")
                dataset_info = gr.Markdown(f"**{default_cfg.label}**\n\n{default_cfg.description}")

                gr.Markdown("**Narrative constraints**")
                temporal_cb = gr.Checkbox(
                    value=default_cfg.supports_temporal,
                    interactive=default_cfg.supports_temporal,
                    label="Constrain narrative temporally",
                )
                citations_cb = gr.Checkbox(
                    value=False, interactive=default_cfg.supports_citations,
                    label="Include citation PageRank (λ)",
                )
                lam_sl = gr.Slider(
                    minimum=0.0, maximum=5.0, value=1.5, step=0.1,
                    label="λ (citation weight)",
                    interactive=default_cfg.supports_citations,
                )
                topk_sl = gr.Slider(
                    minimum=100, maximum=2000, value=pipeline.TOP_K, step=100,
                    label="Top-k candidate set",
                )

            # ---- Query + results column ----------------------------------- #
            with gr.Column(scale=2):
                query_tb = gr.Textbox(
                    value=default_cfg.example_queries[0], label="Your query", lines=3)
                example_dd = gr.Dropdown(
                    choices=list(default_cfg.example_queries),
                    value=default_cfg.example_queries[0],
                    label="Example queries", interactive=True)
                with gr.Row():
                    run_btn = gr.Button("Extract narrative", variant="primary")
                    landscape_btn = gr.Button("Show landscape")

                decomposition_md = gr.Markdown()
                endpoints_md = gr.Markdown()
                gr.Markdown("### 3 · Narrative landscape")
                gr.Markdown(
                    "Grey = all documents · darker grey = top-k candidates · "
                    "colored = clusters the path traverses · black line = the extracted narrative."
                )
                landscape_plot = gr.Plot(label="Landscape")
                metrics_md = gr.Markdown()
                table_df = gr.Dataframe(interactive=False, wrap=True, label="Narrative path")
                summary_md = gr.Markdown()

        # ---- Events -------------------------------------------------------- #
        dataset_dd.change(
            on_dataset_change,
            inputs=dataset_dd,
            outputs=[dataset_info, temporal_cb, citations_cb, lam_sl, example_dd, query_tb],
        )
        example_dd.change(lambda q: q, inputs=example_dd, outputs=query_tb)

        run_btn.click(
            run_pipeline,
            inputs=[dataset_dd, temporal_cb, citations_cb, lam_sl, topk_sl, query_tb],
            outputs=[decomposition_md, endpoints_md, landscape_plot, metrics_md,
                     table_df, summary_md],
        )
        landscape_btn.click(render_landscape, inputs=dataset_dd, outputs=landscape_plot)

    return demo


demo = build_demo()
demo.queue()
if __name__ == "__main__":
    demo.launch()

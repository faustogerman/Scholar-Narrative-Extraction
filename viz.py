"""Altair landscape visualization for the narrative demo.

Reproduces the layered "narrative landscape" figures in the
paper: a grey background of all documents, the top-k candidates, the
clusters the path passes through, and the ordered narrative path itself with
Source/Target labels.
"""

import altair as alt
import numpy as np
import pandas as pd

TOOLTIP = ["title", "abstract", "date", "cluster"]


def _cluster_domain(vis_data: pd.DataFrame) -> list:
    return sorted(vis_data["cluster"].unique(), key=lambda c: (c == "-1", int(c)))


def full_landscape_chart(vis_data: pd.DataFrame, width: int = 760,
                         height: int = 560) -> alt.Chart:
    """The full embedding landscape, colored by HDBSCAN cluster."""
    alt.data_transformers.disable_max_rows()
    domain = _cluster_domain(vis_data)
    color = alt.Color(
        "cluster:N",
        scale=alt.Scale(domain=domain, scheme="category20"),
        legend=None,
    )
    return alt.Chart(vis_data).mark_circle(size=18, opacity=0.6).encode(
        x=alt.X("x:Q", title=None, scale=alt.Scale(zero=False),
                axis=alt.Axis(labels=False, ticks=False, grid=False)),
        y=alt.Y("y:Q", title=None, scale=alt.Scale(zero=False),
                axis=alt.Axis(labels=False, ticks=False, grid=False)),
        color=color,
        tooltip=TOOLTIP,
    ).properties(width=width, height=height).interactive()


def narrative_landscape_chart(vis_data: pd.DataFrame, path: list[int],
                              top_k_indices: np.ndarray, width: int = 760,
                              height: int = 560) -> alt.Chart:
    """Layered figure: background + top-k + path clusters + narrative path."""
    alt.data_transformers.disable_max_rows()

    vis_data = vis_data.copy()
    vis_data["in_topk"] = vis_data["idx"].isin(np.asarray(top_k_indices))

    domain = _cluster_domain(vis_data)
    cluster_color = alt.Color(
        "cluster:N",
        scale=alt.Scale(domain=domain, scheme="category20"),
        legend=None,
    )

    # Clusters the narrative passes through get highlighted in full color.
    path_clusters = set(vis_data.loc[vis_data["idx"].isin(path), "cluster"])
    in_path_cluster = vis_data["cluster"].isin(path_clusters)

    path_df = vis_data.set_index("idx").loc[path].reset_index()
    path_df["step"] = np.arange(len(path_df))

    labels_df = pd.DataFrame({
        "x": [path_df["x"].iloc[0], path_df["x"].iloc[-1]],
        "y": [path_df["y"].iloc[0], path_df["y"].iloc[-1]],
        "label": ["Source", "Target"],
    })

    # Square, path-centered zoom so the narrative fills the canvas.
    pad_frac = 0.15
    x_min, x_max = path_df["x"].min(), path_df["x"].max()
    y_min, y_max = path_df["y"].min(), path_df["y"].max()
    x_cen, y_cen = (x_min + x_max) / 2, (y_min + y_max) / 2
    half = max(x_max - x_min, y_max - y_min) / 2 * (1 + pad_frac)
    # Guard against a zero-span path (single-cluster, tightly packed points).
    half = max(half, 1e-3)

    Xz = alt.X("x:Q", title=None, scale=alt.Scale(domain=[x_cen - half, x_cen + half], clamp=True),
               axis=alt.Axis(labels=False, ticks=False, grid=False))
    Yz = alt.Y("y:Q", title=None, scale=alt.Scale(domain=[y_cen - half, y_cen + half], clamp=True),
               axis=alt.Axis(labels=False, ticks=False, grid=False))

    bg = alt.Chart(vis_data[~vis_data["in_topk"]]).mark_circle(clip=True, size=18).encode(
        Xz, Yz, color=alt.value("#EEEEEE"), tooltip=TOOLTIP)

    topk = alt.Chart(vis_data[vis_data["in_topk"]]).mark_circle(
        clip=True, size=22, stroke="#4a4a4a", strokeWidth=0.4).encode(
        Xz, Yz, color=alt.value("#999999"), tooltip=TOOLTIP)

    clusters_layer = alt.Chart(vis_data[in_path_cluster]).mark_circle(
        clip=True, size=28, stroke="#4a4a4a", strokeWidth=0.4).encode(
        Xz, Yz, color=cluster_color, tooltip=TOOLTIP)

    path_layer = alt.Chart(path_df).mark_line(
        clip=True,
        point=alt.OverlayMarkDef(color="black", size=120, stroke="#ffffff", strokeWidth=1.5),
        color="black", strokeWidth=3.5, strokeCap="round", strokeJoin="round",
    ).encode(Xz, Yz, order=alt.Order("step:Q", sort="ascending"), tooltip=TOOLTIP)

    src_df = labels_df[labels_df["label"] == "Source"]
    tgt_df = labels_df[labels_df["label"] == "Target"]

    def _label(df, align, dx):
        halo = alt.Chart(df).mark_text(
            clip=True, align=align, dx=dx, baseline="middle",
            fontSize=18, fontWeight="bold", stroke="white", strokeWidth=4,
        ).encode(Xz, Yz, text="label:N")
        text = alt.Chart(df).mark_text(
            clip=True, align=align, dx=dx, baseline="middle",
            fontSize=18, fontWeight="bold", color="black",
        ).encode(Xz, Yz, text="label:N")
        return halo + text

    chart = (
        bg + topk + clusters_layer + path_layer
        + _label(src_df, "right", -18) + _label(tgt_df, "left", 18)
    ).properties(width=width, height=height).interactive()
    return chart

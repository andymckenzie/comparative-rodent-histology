"""
cell_shape_classifier.py

Morphological classification of cell shapes from polygon vertex data.

Pipeline:
  1. Load vertex data (per-cell list of (x, y) points)
  2. Clean/validate each polygon
  3. Compute rotation/translation/scale-invariant shape descriptors
  4. Cluster cells into morphological classes (KMeans by default)
  5. (Optional) Apply simple rule-based labels for interpretability
  6. Visualize the classified cells

Author: generated for cell morphology analysis
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon
from shapely.validation import make_valid
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# --------------------------------------------------------------------------
# Pixel-to-micron calibration
# --------------------------------------------------------------------------
# From QuPath's Image tab: Pixel width / height = 0.2626 um (40x, uint8 RGB).
# Update this if you re-calibrate or use a different objective/scan setting.
MICRONS_PER_PIXEL = 0.2626

# Fixed colors for shape categories, so a color always means the same thing
# across every figure -- even if a given file has zero cells in some
# category, "round" is always red, "elongated" is always blue, etc.
SHAPE_LABEL_COLORS = {
    "round": "#E41A1C",             # vivid red
    "elongated": "#377EB8",         # medium blue -- more saturated than steel blue for better contrast against green
    "partial detection": "#4DAF4A", # vivid green -- clearly distinct hue from blue, not just darker/lighter
    "other": "#FF7F00",             # orange (pure yellow is nearly invisible on white; orange stays visible and distinct from red)
}


# --------------------------------------------------------------------------
# 1. Data loading
# --------------------------------------------------------------------------

def load_vertices_from_csv(path: str, id_col="cell_id", x_col="x", y_col="y") -> dict[str, list[tuple[float, float]]]:
    """
    Load vertex data from a long-format CSV:
        cell_id, x, y
        cell_1, 0.0, 0.0
        cell_1, 2.0, 0.0
        cell_1, 2.0, 2.0
        cell_2, ...

    Returns a dict {cell_id: [(x, y), ...]} with vertex order preserved
    from the file (make sure your source data lists vertices in
    perimeter order, not scattered).
    """
    df = pd.read_csv(path)
    cells = {}
    for cid, group in df.groupby(id_col, sort=False):
        cells[str(cid)] = list(zip(group[x_col], group[y_col]))
    return cells


def load_vertices_from_indexed_csv(
    path: str,
    id_col: str = "ObjectID",
    order_col: str = "VertexIndex",
    x_col: str = "X",
    y_col: str = "Y",
    class_col: str | None = "Class",
) -> tuple[dict[str, list[tuple[float, float]]], pd.Series | None]:
    """
    Load vertex data from a long-format CSV that includes an explicit
    per-vertex ordering column, e.g.:

        ObjectID, Class, VertexIndex, X, Y
        86f8...,  Negative, 0, 92688.14, 19711.15
        86f8...,  Negative, 1, 92688.94, 19714.88
        ...

    Sorting by `order_col` guarantees vertices are in perimeter order even
    if rows in the file are out of order. If `class_col` is provided and
    present, also returns a per-cell Series of class labels (useful if
    your data already has a Positive/Negative or similar annotation you
    want to compare against the morphological clusters).

    Returns (cells, class_labels_or_None)
    """
    df = pd.read_csv(path)
    df = df.sort_values([id_col, order_col])

    cells = {}
    for cid, group in df.groupby(id_col, sort=False):
        cells[str(cid)] = list(zip(group[x_col], group[y_col]))

    class_labels = None
    if class_col is not None and class_col in df.columns:
        class_labels = df.groupby(id_col)[class_col].first()
        class_labels.index = class_labels.index.astype(str)

    return cells, class_labels


def load_vertices_from_folder(
    folder_path: str,
    id_col: str = "ObjectID",
    order_col: str = "VertexIndex",
    x_col: str = "X",
    y_col: str = "Y",
    class_col: str | None = "Class",
    pattern: str = "*.csv",
) -> tuple[dict[str, list[tuple[float, float]]], pd.Series | None]:
    """
    Batch-load every CSV file matching `pattern` in a folder (e.g. a local
    path, a mapped network drive like "Z:\\Rat Cell Polygon Vertices Data",
    or a UNC path like r"\\\\servercryo\\StorageCryo\\...\\Rat Cell Polygon
    Vertices Data" -- run this on a machine that actually has access to
    that share; this tool's sandbox cannot reach your network directly).

    Each cell_id is prefixed with its source filename (minus extension) to
    avoid collisions between files that reuse the same ObjectID/UUID space,
    e.g. "RAT_61_LHE_cell_polygons::86f83d9e-...".

    Returns (cells, class_labels_or_None), same shape as
    load_vertices_from_indexed_csv, but pooled across every file found.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(
            f"'{folder_path}' is not a reachable directory from this environment. "
            "If this is a network share (UNC path or mapped drive), run this "
            "script on a machine that has access to it."
        )

    csv_files = sorted(folder.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in '{folder_path}'.")

    all_cells: dict[str, list[tuple[float, float]]] = {}
    all_class_labels: dict[str, str] = {}

    for csv_path in csv_files:
        stem = csv_path.stem
        try:
            cells, class_labels = load_vertices_from_indexed_csv(
                str(csv_path), id_col=id_col, order_col=order_col,
                x_col=x_col, y_col=y_col, class_col=class_col,
            )
        except Exception as e:
            print(f"Skipping '{csv_path.name}' (failed to parse: {e})")
            continue

        for cid, pts in cells.items():
            all_cells[f"{stem}::{cid}"] = pts
        if class_labels is not None:
            for cid, lab in class_labels.items():
                all_class_labels[f"{stem}::{cid}"] = lab

        print(f"Loaded {len(cells)} cells from '{csv_path.name}'")

    class_series = pd.Series(all_class_labels) if all_class_labels else None
    print(f"\nTotal: {len(all_cells)} cells from {len(csv_files)} file(s).")
    return all_cells, class_series


def load_vertices_from_json(path: str) -> dict[str, list[tuple[float, float]]]:
    """
    Load from JSON of the form:
        {"cell_1": [[0,0],[2,0],[2,2],[0,2]], "cell_2": [...]}
    """
    with open(path) as f:
        raw = json.load(f)
    return {cid: [tuple(pt) for pt in pts] for cid, pts in raw.items()}


# --------------------------------------------------------------------------
# 2. Polygon cleaning
# --------------------------------------------------------------------------

def build_clean_polygon(points: Sequence[tuple[float, float]]) -> Polygon | None:
    """
    Build a shapely Polygon from raw vertices, fixing common issues:
      - fewer than 3 points -> invalid, return None
      - self-intersections -> attempt repair via buffer(0)/make_valid
      - duplicate/near-duplicate consecutive points -> collapse them
    """
    pts = np.asarray(points, dtype=float)

    # collapse consecutive near-duplicate points
    if len(pts) > 1:
        keep = [0]
        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i] - pts[keep[-1]]) > 1e-9:
                keep.append(i)
        pts = pts[keep]

    if len(pts) < 3:
        return None

    poly = Polygon(pts)

    if not poly.is_valid:
        poly = make_valid(poly)
        # make_valid can return a GeometryCollection/MultiPolygon; take the
        # largest polygonal piece if so.
        if poly.geom_type != "Polygon":
            polys = [g for g in getattr(poly, "geoms", []) if g.geom_type == "Polygon"]
            if not polys:
                return None
            poly = max(polys, key=lambda g: g.area)

    if poly.area == 0:
        return None

    return poly


# --------------------------------------------------------------------------
# 3. Shape descriptors (rotation/translation/scale invariant where possible)
# --------------------------------------------------------------------------

def min_bounding_rect_dims(poly: Polygon) -> tuple[float, float]:
    """Return (long_side, short_side) of the minimum-area bounding rectangle."""
    mbr = poly.minimum_rotated_rectangle
    if mbr.geom_type != "Polygon":
        # degenerate (near-collinear points)
        minx, miny, maxx, maxy = poly.bounds
        w, h = maxx - minx, maxy - miny
        return max(w, h), max(min(w, h), 1e-9)
    coords = list(mbr.exterior.coords)
    # A rectangle has 4 edges but only 2 unique lengths (opposite sides
    # match). The first two consecutive edges are perpendicular to each
    # other, so they directly give the two distinct side lengths -- do NOT
    # sort all 4 edge lengths and take the top 2, since that just grabs the
    # same (larger) value twice when duplicates are present.
    edge_01 = np.linalg.norm(np.array(coords[1]) - np.array(coords[0]))
    edge_12 = np.linalg.norm(np.array(coords[2]) - np.array(coords[1]))
    long_side, short_side = max(edge_01, edge_12), max(min(edge_01, edge_12), 1e-9)
    return long_side, short_side


def polygon_interior_angles(poly: Polygon) -> np.ndarray:
    """Interior angles (degrees) at each vertex of the polygon exterior."""
    coords = np.array(poly.exterior.coords[:-1])  # drop repeated last point
    n = len(coords)
    angles = np.zeros(n)
    for i in range(n):
        p_prev = coords[i - 1]
        p_curr = coords[i]
        p_next = coords[(i + 1) % n]
        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angles[i] = np.degrees(np.arccos(cos_angle))
    return angles


def shape_features(poly: Polygon, angle_deviation_thresh: float = 15.0, microns_per_pixel: float = MICRONS_PER_PIXEL) -> dict:
    """
    Compute a feature vector describing overall morphology of a cell polygon.
    The dimensionless descriptors (circularity, solidity, aspect_ratio, etc.)
    are scale-normalized so cell *size* does not drive clustering. Alongside
    these, real physical measurements in micrometers (um) are also included
    -- these use `microns_per_pixel` to convert from raw pixel coordinates,
    so make sure it matches your imaging calibration (see MICRONS_PER_PIXEL
    at the top of this file, e.g. from QuPath's Image tab).
    """
    area = poly.area
    perimeter = poly.length
    hull = poly.convex_hull

    circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-12)          # 1.0 = circle
    solidity = area / (hull.area + 1e-12)                              # concavity measure
    long_side, short_side = min_bounding_rect_dims(poly)
    aspect_ratio = long_side / short_side                              # elongation
    rectangularity = area / (long_side * short_side + 1e-12)

    # Count meaningful corners while filtering near-collinear vertices.
    # Retained because this was an input to the original KMeans model.
    angles = polygon_interior_angles(poly)
    n_true_corners = int(np.sum(np.abs(angles - 180.0) > angle_deviation_thresh))
    # normalized perimeter (isoperimetric-style, scale-free): compactness
    compactness = perimeter / (2 * np.sqrt(np.pi * area) + 1e-12)      # 1.0 = circle, higher = more irregular

    # --- real-world size measurements, converted to micrometers ---
    mpp = microns_per_pixel
    long_side_um = long_side * mpp
    short_side_um = short_side * mpp
    perimeter_um = perimeter * mpp
    area_um2 = area * (mpp ** 2)   # area scales with the square of the linear factor
    equivalent_diameter_um = 2 * np.sqrt(area_um2 / np.pi)  # diameter of a circle with the same area

    return {
        "n_vertices_raw": len(poly.exterior.coords) - 1,
        "n_true_corners": n_true_corners,
        "area": area,
        "perimeter": perimeter,
        "circularity": circularity,
        "solidity": solidity,
        "aspect_ratio": aspect_ratio,
        "rectangularity": rectangularity,
        "compactness": compactness,
        "long_axis_um": long_side_um,
        "short_axis_um": short_side_um,
        "perimeter_um": perimeter_um,
        "area_um2": area_um2,
        "equivalent_diameter_um": equivalent_diameter_um,
    }


def build_feature_table(cells: dict[str, list[tuple[float, float]]], microns_per_pixel: float = MICRONS_PER_PIXEL) -> pd.DataFrame:
    """Build a DataFrame of features, one row per valid cell."""
    rows = []
    for cid, pts in cells.items():
        poly = build_clean_polygon(pts)
        if poly is None:
            continue
        feats = shape_features(poly, microns_per_pixel=microns_per_pixel)
        feats["cell_id"] = cid
        rows.append(feats)
    df = pd.DataFrame(rows).set_index("cell_id")
    return df


# --------------------------------------------------------------------------
# 4. Clustering (unsupervised morphological classification)
# --------------------------------------------------------------------------

# Features that describe SHAPE (not size). Keep area/perimeter out unless
# you want size to influence class membership.
DEFAULT_CLUSTER_FEATURES = [
    "circularity",
    "solidity",
    "aspect_ratio",
    "rectangularity",
    "compactness",
    "n_true_corners",
]


def choose_k_by_silhouette(X: np.ndarray, k_range=range(2, 8)) -> tuple[int, dict[int, float]]:
    """Try several k values, return the best k and all silhouette scores."""
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = km.fit_predict(X)
        scores[k] = silhouette_score(X, labels)
    best_k = max(scores, key=scores.get)
    return best_k, scores


def cluster_shapes(
    df: pd.DataFrame,
    features: list[str] = DEFAULT_CLUSTER_FEATURES,
    n_clusters: int | None = None,
) -> pd.DataFrame:
    """
    Cluster cells into morphological classes using KMeans on standardized
    shape descriptors. If n_clusters is None, pick the best k automatically
    via silhouette score (tested over k=2..7).
    """
    X = df[features].to_numpy()
    X_scaled = StandardScaler().fit_transform(X)

    if n_clusters is None:
        n_clusters, scores = choose_k_by_silhouette(X_scaled)
        print(f"Auto-selected k={n_clusters} (silhouette scores: {scores})")

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels = km.fit_predict(X_scaled)

    out = df.copy()
    out["cluster"] = labels
    return out


# --------------------------------------------------------------------------
# 5. Optional: simple rule-based labels for interpretability
#    (useful as human-readable names layered on top of clusters, or as a
#    standalone classifier if you don't want unsupervised clustering)
# --------------------------------------------------------------------------

def rule_based_label(
    row: pd.Series,
    concave_solidity_thresh: float = 0.75,
    elongated_aspect_ratio_thresh: float = 1.5,
    extreme_compactness_thresh: float = 1.6,
) -> str:
    """
    Assign a human-readable shape category based on descriptor thresholds.

    Restructured so ROUND is a broad catch-all rather than requiring high
    circularity: a cell that is compact (not concave), not elongated, and
    not extremely jagged is called round even if a few sharp local peaks
    on its boundary pull its raw circularity down. Circularity alone
    over-penalizes minor boundary noise that doesn't actually change the
    overall roundness of the shape.

    "other" is now reserved only for genuinely extreme outlines --
    compactness (perimeter relative to an equal-area circle) above
    extreme_compactness_thresh flags the ~2% most jagged/irregular shapes
    in a typical batch, e.g. star-like or highly convoluted boundaries that
    go well beyond "a few sharp peaks." Tune this value up to shrink
    "other" further (stricter, fewer flagged) or down to widen it.

    Order of checks (first match wins):
      1. partial detection: solidity < concave_solidity_thresh (genuinely concave/crescent)
      2. other: compactness > extreme_compactness_thresh (truly extreme jaggedness)
      3. elongated: aspect_ratio > elongated_aspect_ratio_thresh
      4. round: everything else (the catch-all)
    """
    if row["solidity"] < concave_solidity_thresh:
        return "partial detection"
    if row["compactness"] > extreme_compactness_thresh:
        return "other"
    if row["aspect_ratio"] > elongated_aspect_ratio_thresh:
        return "elongated"
    return "round"


def add_rule_based_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rule_label"] = out.apply(rule_based_label, axis=1)
    return out


# --------------------------------------------------------------------------
# 6. Visualization
# --------------------------------------------------------------------------

def plot_classified_cells(
    cells: dict[str, list[tuple[float, float]]],
    df: pd.DataFrame,
    label_col: str = "rule_label",
    title: str = "Cell shape classification",
    save_path: str | None = None,
    show: bool = True,
    microns_per_pixel: float = MICRONS_PER_PIXEL,
):
    """
    Plot every cell polygon, colored by its assigned class. Defaults to
    `rule_label` (round / elongated / partial detection / other)
    since that's human-readable; pass label_col="cluster" for the raw
    KMeans cluster numbers instead.

    Coordinates are converted from pixels to micrometers using
    `microns_per_pixel` so the axes show real physical scale, and a scale
    bar is drawn in the corner as a visual reference.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    labels = df[label_col]
    unique_labels = sorted(labels.unique(), key=str)
    cmap = plt.get_cmap("tab10")
    fallback_idx = 0
    color_map = {}
    for lab in unique_labels:
        if lab in SHAPE_LABEL_COLORS:
            color_map[lab] = SHAPE_LABEL_COLORS[lab]
        else:
            # not a recognized shape category (e.g. numeric cluster IDs) --
            # fall back to a standard colormap so those still get distinct colors
            color_map[lab] = cmap(fallback_idx % 10)
            fallback_idx += 1

    patches = []
    colors = []
    all_pts_um = []
    for cid, pts in cells.items():
        if cid not in df.index:
            continue
        poly_pts_um = np.array(pts) * microns_per_pixel
        all_pts_um.append(poly_pts_um)
        patches.append(MplPolygon(poly_pts_um, closed=True))
        colors.append(color_map[labels.loc[cid]])

    collection = PatchCollection(patches, facecolor=colors, linewidths=0.3)
    collection.set_edgecolor((0, 0, 0, 0.35))  # semi-transparent outline so fill colors stay dominant
    ax.add_collection(collection)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel("WSI X coordinate (µm)")
    ax.set_ylabel("WSI Y coordinate (µm)")
    # Title intentionally omitted; filename identifies the figure.
    ax.invert_yaxis()  # image coordinates have Y increasing downward (origin top-left),
                       # matplotlib defaults to Y increasing upward -- invert to match
                       # how the tissue actually looks in QuPath / the source image

    # legend
    for lab, color in color_map.items():
        ax.plot([], [], "s", color=color, label=str(lab))
    ax.legend(loc="upper right", title="Classification")

    # scale bar: pick a round number roughly 1/6th of the plot width
    if all_pts_um:
        all_pts_um = np.concatenate(all_pts_um, axis=0)
        x_range = all_pts_um[:, 0].max() - all_pts_um[:, 0].min()
        raw_len = x_range / 6
        magnitude = 10 ** np.floor(np.log10(raw_len)) if raw_len > 0 else 1
        for mult in (1, 2, 5, 10):
            bar_len = mult * magnitude
            if bar_len >= raw_len:
                break
        x0 = all_pts_um[:, 0].min() + x_range * 0.05
        y0 = all_pts_um[:, 1].min() - (all_pts_um[:, 1].max() - all_pts_um[:, 1].min()) * 0.06
        ax.plot([x0, x0 + bar_len], [y0, y0], color="black", linewidth=3, solid_capstyle="butt", clip_on=False)
        ax.annotate(f"{bar_len:.0f} µm", (x0 + bar_len / 2, y0), xytext=(0, -7), textcoords="offset points", ha="center", va="top", fontsize=9)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# --------------------------------------------------------------------------
# 7. Example usage / synthetic demo
# --------------------------------------------------------------------------

def _make_synthetic_cells(n_per_type: int = 15, seed: int = 0) -> dict[str, list[tuple[float, float]]]:
    """Generate synthetic cells of a few morphological types, for demoing
    the pipeline without needing real data."""
    rng = np.random.default_rng(seed)
    cells = {}
    idx = 0

    def jitter(pts, scale=0.08):
        pts = np.array(pts, dtype=float)
        return pts + rng.normal(0, scale, pts.shape)

    # round cells (regular polygons with many sides, approximating circles)
    for _ in range(n_per_type):
        n = rng.integers(10, 16)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        r = rng.uniform(0.9, 1.1)
        pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        cells[f"round_{idx}"] = jitter(pts).tolist()
        idx += 1

    # elongated / spindle-shaped cells
    for _ in range(n_per_type):
        n = rng.integers(6, 10)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        rx, ry = rng.uniform(1.8, 2.4), rng.uniform(0.5, 0.7)
        pts = np.column_stack([rx * np.cos(angles), ry * np.sin(angles)])
        cells[f"elongated_{idx}"] = jitter(pts).tolist()
        idx += 1

    # irregular/concave (star-like) cells
    for _ in range(n_per_type):
        n = rng.integers(6, 10)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        r = np.where(np.arange(n) % 2 == 0, rng.uniform(1.0, 1.2), rng.uniform(0.3, 0.5))
        pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        cells[f"irregular_{idx}"] = jitter(pts, scale=0.03).tolist()
        idx += 1

    return cells


def process_folder_per_file(
    folder_path: str,
    output_dir: str = "classified_output",
    pattern: str = "*.csv",
    n_clusters: int | None = None,
    show_plots: bool = False,
    microns_per_pixel: float = MICRONS_PER_PIXEL,
) -> dict[str, pd.DataFrame]:
    """
    Process every CSV in a folder INDEPENDENTLY: each file gets its own
    feature table, its own clustering run, its own rule-based labels, and
    its own saved CSV + plots. This keeps each rat/sample's shape
    classification self-contained instead of pooling every file into one
    combined population (which is what load_vertices_from_folder +
    a single cluster_shapes call would do).

    Results are written to `output_dir`, one classified CSV and two PNGs
    per input file, named after the original file's stem.

    Returns a dict {file_stem: classified_dataframe} for all files that
    parsed successfully, in case you want to inspect results in-session
    without re-reading from disk.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(
            f"'{folder_path}' is not a reachable directory from this environment. "
            "If this is a network share (UNC path or mapped drive), run this "
            "script on a machine that has access to it."
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(folder.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in '{folder_path}'.")

    results = {}

    for csv_path in csv_files:
        stem = csv_path.stem
        print(f"\n=== Processing '{csv_path.name}' ===")

        try:
            cells, class_labels = load_vertices_from_indexed_csv(str(csv_path))
        except Exception as e:
            print(f"  Skipped (failed to parse: {e})")
            continue

        df = build_feature_table(cells, microns_per_pixel=microns_per_pixel)
        if len(df) == 0:
            print("  Skipped (no valid polygons found).")
            continue
        print(f"  {len(df)} valid cells out of {len(cells)} total.")

        # Guard against files with too few cells to cluster meaningfully.
        k = n_clusters
        if k is None:
            max_k = min(7, len(df) - 1)
            if max_k < 2:
                print("  Too few cells to cluster (need >= 3); skipping clustering.")
                df["cluster"] = 0
            else:
                df = cluster_shapes(df, n_clusters=None if max_k >= 2 else 2)
        else:
            df = cluster_shapes(df, n_clusters=k)

        df = add_rule_based_labels(df)
        if class_labels is not None:
            df["source_class"] = class_labels.reindex(df.index)

        # Save this file's results
        csv_out = out_dir / f"{stem}_classified.csv"
        df.to_csv(csv_out)
        print(f"  Saved: {csv_out}")
        print(f"  Mean equivalent diameter: {df['equivalent_diameter_um'].mean():.2f} um | "
              f"Mean long axis: {df['long_axis_um'].mean():.2f} um")

        cluster_plot_path = out_dir / f"{stem}_clusters.png"
        plot_classified_cells(
            cells, df, label_col="cluster",
            title=f"{stem} - KMeans clusters",
            save_path=str(cluster_plot_path),
            show=show_plots,
            microns_per_pixel=microns_per_pixel,
        )

        rule_plot_path = out_dir / f"{stem}_shape_labels.png"
        plot_classified_cells(
            cells, df, label_col="rule_label",
            title=f"{stem} - shape labels",
            save_path=str(rule_plot_path),
            show=show_plots,
            microns_per_pixel=microns_per_pixel,
        )

        results[stem] = df

    print(f"\nDone. Processed {len(results)} file(s) into '{out_dir}'.")
    return results


def main():
    # --- Per-file batch mode: point at a folder of CSVs (e.g. your network
    # share). Each file is analyzed and classified INDEPENDENTLY -- results
    # are written to `output_dir`, one classified CSV + PNG per file.
    #   folder = r"\\servercryo\StorageCryo\Research\Rat Research\Rat Histology Quant\Rat Cell Polygon Vertices Data"
    #   folder = r"Z:\Rat Histology Quant\Rat Cell Polygon Vertices Data"  # if mapped to Z:
    folder = r"\\servercryo\StorageCryo\Research\Rat Research\Rat Histology Quant\Rat Cell Polygon Vertices Data"
    output_dir = r"C:\Users\SarahSe\Desktop\Rats\Figures"

    try:
        results = process_folder_per_file(folder, output_dir=output_dir, show_plots=False)
    except FileNotFoundError as e:
        print(f"[folder mode unavailable here: {e}]")
        print("Falling back to single local file for this demo run.")
        cells, class_labels = load_vertices_from_indexed_csv("RAT_61_LHE_cell_polygons.csv")

        df = build_feature_table(cells)
        print(f"Loaded {len(df)} valid cells out of {len(cells)} total.")

        df = cluster_shapes(df, n_clusters=None)
        df = add_rule_based_labels(df)

        print(df[["circularity", "solidity", "aspect_ratio", "n_true_corners", "cluster", "rule_label"]])
        print("\nCluster vs rule-based label cross-tab:")
        print(pd.crosstab(df["cluster"], df["rule_label"]))

        plot_classified_cells(cells, df, label_col="cluster", title="KMeans morphological clusters")
        plot_classified_cells(cells, df, label_col="rule_label", title="Rule-based shape labels")
        return {"RAT_61_LHE_cell_polygons": df}

    return results


if __name__ == "__main__":
    main()


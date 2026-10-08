#!/usr/bin/env python3
"""
cross_reference_GL_DB.py

Identify rare GL_DB germline-database variants in a TSOPPI run and detect
cross-sample contamination by comparing low-VAF variants in each sample against
all GL_DB variants in every other sample.

Algorithm
---------
1. Per-sample: retain GL_DB variants with AF_TUMOR <= VAF_th  (reference set)
2. Per-sample: load all GL_DB variants regardless of VAF       (query set)
3. Cross-reference: for each (ref, query) pair, look up ref's filtered variants
   in query's complete data — no second VAF filter is applied on lookup
4. Report intersection counts and visualise as heatmap + VAF matrix

This design exposes cases where a variant is rare in one sample (low VAF) but
present at any frequency in another, which is the contamination signal of interest.

Usage
-----
    uv run cross_reference_GL_DB.py \\
        --TSOPPI_folder PATH        \\
        [--alias NAME]              \\
        [--VAF_th 0.02]             \\
        [--output_dir PATH]         \\
        [--tmp]

Outputs (written to output_dir, default: TSOPPI_folder)
-------------------------------------------------------
    {run}_GL_DB_cross_reference_table.tsv   long-format shared-variant table
    {run}_GL_DB_intersection_matrix.tsv     N x N count matrix
    {run}_GL_DB_intersection_heatmap.pdf    barplot + intersection heatmap
    {run}_GL_DB_variant_sample_heatmap.pdf  VAF heatmap (overlap variants only)
    {sample}_GL_DB_filtered_variants.tsv    per-sample filtered tables (--tmp)
"""
import argparse
import sys
from pathlib import Path

import polars as pl

# Columns retained throughout the analysis
_VARIANT_COLS = ["TAL", "AF_TUMOR", "GENOMIC_CHANGE", "SYMBOL", "VCF_SAMPLE_ID"]
_XREF_COLS    = ["TAL", "AF_TUMOR", "GENOMIC_CHANGE", "SYMBOL", "sampleID_reference", "sampleID_query"]


# ----------------------------- #
# Command-line argument parsing # 
# ----------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-reference rare GL_DB variants across samples in a TSOPPI run folder."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--alias",
        type=str,
        default="cross-ref_GL_DB",
        metavar="STR",
        help="Alias/name for this analysis run. Used in plot title and output file names.",
    )
    parser.add_argument(
        "--VAF_th",
        type=float,
        default=0.02,
        metavar="FLOAT",
        help="Maximum AF_TUMOR threshold for variant selection. "
             "Variants with missing AF_TUMOR are excluded.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory for all output files. Created if absent. "
             "Defaults to TSOPPI_folder.",
    )
    parser.add_argument(
        "--tmp",
        action="store_true",
        default=False,
        help="Also export per-sample filtered variant tables as TSV files.",
    )
    parser.add_argument(
        "--TSOPPI_folder",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to the TSOPPI run folder containing per-sample subdirectories.",
    )
    return parser.parse_args()


# Revisado hasta aqui


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_pcgr_files(run_folder: Path) -> dict[str, Path]:
    """
    Walk immediate subdirectories of run_folder and locate the PCGR tiers TSV
    for each sample.  Directories ending with '_metrics_plots' are ignored.

    Returns
    -------
    dict mapping sample_id -> absolute Path of the tiers TSV.
    """
    pcgr_files: dict[str, Path] = {}

    for sample_dir in sorted(run_folder.iterdir()):
        if not sample_dir.is_dir():
            continue
        if sample_dir.name.endswith("_metrics_plots"):
            continue

        sample_id = sample_dir.name
        # Glob is robust to PCGR version differences in the folder name.
        matches = sorted(
            sample_dir.glob(
                f"{sample_id}_PCGR_*_all/{sample_id}.pcgr_acmg.grch37.snvs_indels.tiers.tsv"
            )
        )

        if not matches:
            print(
                f"[WARNING] No PCGR tiers TSV found for sample '{sample_id}' — skipping.",
                file=sys.stderr,
            )
            continue

        if len(matches) > 1:
            print(
                f"[WARNING] Multiple PCGR tiers TSVs for '{sample_id}'; using: {matches[0]}",
                file=sys.stderr,
            )

        pcgr_files[sample_id] = matches[0]

    return pcgr_files


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def load_and_filter(tsv_path: Path, vaf_th: float) -> pl.DataFrame:
    """
    Read a PCGR tiers TSV and return rows that satisfy:
        TAL == "GL_DB"  AND  AF_TUMOR is present  AND  AF_TUMOR <= vaf_th

    Rows with a missing AF_TUMOR value are excluded from analysis.

    Only the columns in _VARIANT_COLS are retained in the result.
    """
    df = pl.read_csv(tsv_path, separator="\t", infer_schema_length=0)

    # String-encoded "NA" → null; then cast to float for numeric comparison.
    df = df.with_columns(
        pl.col("GLOBAL_AF_1KG").replace("NA", None).cast(pl.Float64),
        pl.col("AF_TUMOR").replace("NA", None).cast(pl.Float64),
    )

    return (
        df
        .filter(
            (pl.col("TAL") == "GL_DB")
            & pl.col("AF_TUMOR").is_not_null()
            & (pl.col("AF_TUMOR") <= vaf_th)
        )
        .select(_VARIANT_COLS)
    )


# ---------------------------------------------------------------------------
# Cross-reference
# ---------------------------------------------------------------------------

def load_complete_gl_db_data(pcgr_files: dict[str, Path]) -> dict[str, pl.DataFrame]:
    """
    Load complete GL_DB variants from all samples without VAF threshold filtering.
    
    Returns
    -------
    dict[sample_id] → DataFrame with all GL_DB variants (TAL=="GL_DB", AF_TUMOR present)
    """
    complete_data: dict[str, pl.DataFrame] = {}
    
    for sample_id, tsv_path in pcgr_files.items():
        df = pl.read_csv(tsv_path, separator="\t", infer_schema_length=0)
        
        # Convert NA values
        df = df.with_columns(
            pl.col("AF_TUMOR").replace("NA", None).cast(pl.Float64),
        )
        
        # Filter: TAL == GL_DB AND AF_TUMOR is present (no VAF threshold)
        df_complete = (
            df
            .filter(
                (pl.col("TAL") == "GL_DB")
                & pl.col("AF_TUMOR").is_not_null()
            )
            .select(_VARIANT_COLS)
        )
        
        complete_data[sample_id] = df_complete
    
    return complete_data


# ---------------------------------------------------------------------------
# Cross-reference
# ---------------------------------------------------------------------------

def build_cross_reference_table(
    filtered: dict[str, pl.DataFrame],
    complete: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    """
    For every (reference, query) pair where reference != query, find variants
    from the reference sample's FILTERED set (VAF <= threshold) in the query
    sample's COMPLETE data (all GL_DB variants, no threshold).

    The resulting long-format table replaces VCF_SAMPLE_ID with two columns:
    sampleID_reference and sampleID_query.  Variant-level data (AF_TUMOR, etc.)
    reflects the query sample's observation.
    """
    rows: list[pl.DataFrame] = []

    for ref_id, ref_df in filtered.items():
        if ref_df.is_empty():
            continue

        ref_variants = set(ref_df["GENOMIC_CHANGE"].to_list())

        for query_id, query_df in complete.items():
            if query_id == ref_id or query_df.is_empty():
                continue

            shared = query_df.filter(pl.col("GENOMIC_CHANGE").is_in(ref_variants))
            if shared.is_empty():
                continue

            rows.append(
                shared
                .drop("VCF_SAMPLE_ID")
                .with_columns(
                    pl.lit(ref_id).alias("sampleID_reference"),
                    pl.lit(query_id).alias("sampleID_query"),
                )
                .select(_XREF_COLS)
            )

    if not rows:
        return pl.DataFrame(
            schema={
                "TAL":               pl.Utf8,
                "AF_TUMOR":          pl.Float64,
                "GLOBAL_AF_1KG":     pl.Float64,
                "GENOMIC_CHANGE":    pl.Utf8,
                "SYMBOL":            pl.Utf8,
                "sampleID_reference": pl.Utf8,
                "sampleID_query":    pl.Utf8,
            }
        )

    return pl.concat(rows)


def build_intersection_matrix(
    filtered: dict[str, pl.DataFrame],
    complete: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    """
    Build a sample × sample integer matrix of variant presence counts.
    
    Reference set (rows): variants from filtered data (VAF_th applied)
    Query set (columns): variants from complete data (no VAF_th)
    
    Diagonal: count of filtered variants in that sample
    Off-diagonal[ref, query]: count of ref's filtered variants found in query's complete data

    Parameters
    ----------
    filtered : dict[sample_id] → DataFrame with variants that passed VAF_th per sample
    complete : dict[sample_id] → DataFrame with all GL_DB variants (no VAF_th)
    """
    sample_ids = list(filtered.keys())

    data: dict[str, list[int]] = {"sample": sample_ids}  # type: ignore[assignment]
    
    for query_id in sample_ids:
        counts = []
        query_variants = set(complete[query_id]["GENOMIC_CHANGE"].to_list()) if not complete[query_id].is_empty() else set()
        
        for ref_id in sample_ids:
            ref_variants = set(filtered[ref_id]["GENOMIC_CHANGE"].to_list()) if not filtered[ref_id].is_empty() else set()
            # Count how many ref's filtered variants appear in query's complete data
            counts.append(len(ref_variants & query_variants))
        
        data[query_id] = counts

    return pl.DataFrame(data)


def build_overlap_vaf_matrix(
    filtered: dict[str, pl.DataFrame],
    complete: dict[str, pl.DataFrame],
) -> tuple[list[str], list[str], dict[str, dict[str, float]]]:
    """
    Build a VAF matrix restricted to variants that are relevant overlaps:
    variants that passed VAF_th in at least one reference sample AND appear
    in at least one other sample's complete GL_DB data.

    Returns
    -------
    variant_ids : sorted list of GENOMIC_CHANGE strings
    sample_ids  : sorted list of sample identifiers
    vaf_data    : dict[variant_id][sample_id] = VAF (0.0 if absent in that sample)
    """
    # Collect variants that passed VAF_th in any sample AND appear in another
    relevant: set[str] = set()
    for ref_id, ref_df in filtered.items():
        if ref_df.is_empty():
            continue
        ref_variants = set(ref_df["GENOMIC_CHANGE"].to_list())
        for query_id, query_df in complete.items():
            if query_id == ref_id or query_df.is_empty():
                continue
            relevant.update(ref_variants & set(query_df["GENOMIC_CHANGE"].to_list()))

    variant_ids = sorted(relevant)
    sample_ids = sorted(complete.keys())

    vaf_data: dict[str, dict[str, float]] = {v: {s: 0.0 for s in sample_ids} for v in variant_ids}

    for sample_id, df in complete.items():
        if df.is_empty() or not relevant:
            continue
        per_variant = (
            df.filter(pl.col("GENOMIC_CHANGE").is_in(relevant))
            .select(["GENOMIC_CHANGE", "AF_TUMOR"])
            .group_by("GENOMIC_CHANGE")
            .agg(pl.col("AF_TUMOR").max())
        )
        for row in per_variant.iter_rows(named=True):
            if row["AF_TUMOR"] is not None:
                vaf_data[row["GENOMIC_CHANGE"]][sample_id] = float(row["AF_TUMOR"])

    return variant_ids, sample_ids, vaf_data


def cluster_vaf_matrix(
    vaf_data: dict[str, dict[str, float]],
    variant_ids: list[str],
    sample_ids: list[str],
) -> tuple[list[int], list[int]]:
    """
    Hierarchically cluster both axes of a VAF matrix (Ward linkage, Euclidean distance).

    Returns
    -------
    variant_order : leaf indices to reorder variants
    sample_order  : leaf indices to reorder samples
    """
    import numpy as np
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist

    matrix = np.array(
        [[vaf_data[v][s] for s in sample_ids] for v in variant_ids],
        dtype=float,
    )

    def _leaf_order(data: np.ndarray) -> list[int]:
        if data.shape[0] <= 1:
            return [0]
        return list(dendrogram(linkage(pdist(data, metric="euclidean"), method="ward"), no_plot=True)["leaves"])

    return _leaf_order(matrix), _leaf_order(matrix.T)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_intersection_heatmap(
    matrix: pl.DataFrame,
    sample_counts: dict[str, int],
    output_path: Path,
    run_name: str,
    vaf_th: float,
    alias: str,
) -> None:
    """
    Composite figure: barplot of per-sample rare variant counts (top panel) and
    pairwise intersection heatmap (bottom panel).

    Built directly with matplotlib (plotnine's own backend) so that a single
    combined vector PDF can be produced without an external compositor library.
    Aesthetics follow plotnine / ggplot2 conventions: white panel background,
    light grid lines, white-to-blue continuous fill scale.
    """
    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")
    import matplotlib.gridspec as gridspec
    import numpy as np

    sample_ids = matrix["sample"].to_list()
    n          = len(sample_ids)
    counts     = [sample_counts[s] for s in sample_ids]

    # Numeric matrix  (row i = reference sample i, col j = query sample j)
    mat = np.array(
        [
            [matrix.filter(pl.col("sample") == ref).select(qry).item()
             for qry in sample_ids]
            for ref in sample_ids
        ],
        dtype=float,
    )

    # Mask lower triangle to show only upper triangle (including diagonal)
    # to avoid redundancy in this symmetric matrix
    mask = np.tril(np.ones_like(mat, dtype=bool), k=-1)
    mat_masked = np.ma.array(mat, mask=mask)

    # Color scale based only on off-diagonal values (contamination signals)
    # Diagonal cells will remain white (self-counts)
    off_diag_values = mat[~np.eye(mat.shape[0], dtype=bool)]
    off_diag_values = off_diag_values[off_diag_values > 0]  # exclude zeros
    # Use fixed cap of 50: provides good color differentiation for typical contamination
    # values; any value > 50 saturates to darkest blue (still visible as concerning)
    vmax = max(50, off_diag_values.max()) if len(off_diag_values) > 0 else 50

    # Use ColorBrewer Blues: perceptually uniform, publication-quality palette
    cmap = plt.cm.Blues

    # Figure geometry scales with sample count
    fig_w = max(8.0, n * 2.0)
    fig   = plt.figure(figsize=(fig_w, fig_w * 1.05), facecolor="white")
    # Single-column GridSpec for barplot and heatmap
    gs = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[1, 3],
        hspace=0.2,
        left=0.15, right=0.92, top=0.93, bottom=0.18,
    )
    ax_bar  = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1], sharex=ax_bar)

    # ── Barplot ──────────────────────────────────────────────────────────────
    x_pos = np.arange(n)
    bars  = ax_bar.bar(x_pos, counts, color="#1565C0", alpha=0.85, width=0.65,
                       zorder=3)
    y_nudge = max(counts) * 0.025
    for bar, count in zip(bars, counts):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            count + y_nudge,
            str(count),
            ha="center", va="bottom", fontsize=8.5, color="#1a1a1a",
        )
    ax_bar.set_xlim(-0.5, n - 0.5)
    ax_bar.set_ylim(0, max(counts) * 1.15)
    ax_bar.set_xticks([])
    ax_bar.tick_params(labelbottom=False)
    ax_bar.set_ylabel("Variant count", fontsize=10)
    ax_bar.set_title(
        f"{alias} — GL_DB variants (AF ≤ {vaf_th})",
        fontsize=14, pad=12, loc="center", fontweight="bold",
    )
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.yaxis.grid(True, color="#e0e0e0", linestyle="--", linewidth=0.7,
                      zorder=0)
    ax_bar.set_axisbelow(True)
    ax_bar.set_facecolor("white")

    # ── Heatmap ───────────────────────────────────────────────────────────────
    # imshow: pixel (j, i) = mat[i, j]; row 0 displayed at top by default.
    # Use masked array to show only upper triangle (including diagonal)
    im = ax_heat.imshow(
        mat_masked, cmap=cmap, vmin=0, vmax=vmax,
        aspect="auto", interpolation="nearest",
    )
    # Manually set diagonal cells to white with neutral grey background for visual distinction
    # This highlights diagonal (self-counts) without visual clutter
    for i in range(n):
        ax_heat.add_patch(
            plt.Rectangle((i - 0.5, i - 0.5), 1, 1, 
                         facecolor="#d2cece", edgecolor="none", zorder=1)
        )
    # Cell text with adaptive colour (white on dark cells, charcoal on light)
    # Diagonal cells get bold, larger font on subtle grey background
    for i in range(n):
        for j in range(n):
            if i <= j:  # Upper triangle and diagonal only
                is_diagonal = (i == j)
                
                if is_diagonal:
                    # Diagonal: bold, larger text on grey background
                    ax_heat.text(
                        j, i, str(int(mat[i, j])),
                        ha="center", va="center",
                        fontsize=10, fontweight="bold", color="#1a1a1a",
                    )
                else:
                    # Off-diagonal: standard adaptive text
                    brightness = mat[i, j] / vmax if vmax > 0 else 0
                    tc = "white" if brightness > 0.55 else "#1a1a1a"
                    ax_heat.text(
                        j, i, str(int(mat[i, j])),
                        ha="center", va="center", fontsize=8.5, color=tc,
                    )

    ax_heat.set_xticks(x_pos)
    ax_heat.set_xticklabels(sample_ids, rotation=35, ha="center", fontsize=8)
    # Move x-axis tick labels to top, but keep "Query sample" label at bottom
    ax_heat.xaxis.tick_top()
    ax_heat.xaxis.set_label_position('bottom')
    ax_heat.set_yticks(x_pos)
    ax_heat.set_yticklabels(sample_ids, fontsize=8)
    ax_heat.set_xlabel("Query sample", fontsize=10)
    ax_heat.set_ylabel("Reference sample", fontsize=10)
    ax_heat.tick_params(length=0)
    ax_heat.set_facecolor("white")
    for spine in ax_heat.spines.values():
        spine.set_visible(False)

    # Thin grid lines to separate cells
    for pos in np.arange(n) - 0.5:
        ax_heat.axhline(pos, color="white", linewidth=0.8)
        ax_heat.axvline(pos, color="white", linewidth=0.8)

    fig.savefig(output_path, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Heatmap saved to: {output_path}")


def plot_variant_sample_heatmap(
    vaf_data: dict[str, dict[str, float]],
    variant_ids: list[str],
    sample_ids: list[str],
    variant_order: list[int],
    sample_order: list[int],
    output_path: Path,
    alias: str,
) -> None:
    """
    VAF heatmap: samples (rows) × variants (columns).
    
    Visualizes tumor allele frequencies for GL_DB variants across samples.
    Uses blues colorbrewer palette (white=0, dark blue=high VAF).
    Cell values show exact VAF percentages.
    Both axes are reordered by hierarchical clustering for pattern visualization.
    Shows only variants present in >= 2 samples for improved readability.
    """
    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")
    import numpy as np

    n_variants = len(variant_ids)
    n_samples = len(sample_ids)

    # Build reordered VAF matrix: samples (rows) × variants (columns)
    reordered_variants = [variant_ids[i] for i in variant_order]
    reordered_samples = [sample_ids[j] for j in sample_order]

    vaf_matrix = np.array(
        [[vaf_data[v][s] for v in reordered_variants] for s in reordered_samples],
        dtype=float,
    )

    # Figure geometry: scale with matrix size
    fig_w = max(14.0, n_variants * 0.5 + 2.0)
    fig_h = max(8.0, n_samples * 0.8 + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")

    # ── Heatmap with VAF values ──────────────────────────────────────────
    from matplotlib.patches import Rectangle
    
    cmap = plt.cm.Blues
    im = ax.imshow(
        vaf_matrix, cmap=cmap, vmin=0, vmax=1,
        aspect="auto", interpolation="nearest",
    )
    
    # Draw white rectangles over cells with VAF == 0
    for i in range(n_samples):
        for j in range(n_variants):
            if vaf_matrix[i, j] == 0.0:
                rect = Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                fill=True, facecolor='white', edgecolor='none')
                ax.add_patch(rect)

    # Axis labels and ticks
    ax.set_xticks(np.arange(n_variants))
    ax.set_yticks(np.arange(n_samples))
    ax.set_xticklabels(reordered_variants, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(reordered_samples, fontsize=9)

    ax.set_xlabel("Variant", fontsize=10, fontweight="bold")
    ax.set_ylabel("Sample ID", fontsize=10, fontweight="bold")

    # Grid lines for cell delineation
    ax.set_xticks(np.arange(n_variants) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_samples) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)

    # Add text annotations with VAF percentages
    for i in range(n_samples):
        for j in range(n_variants):
            vaf = vaf_matrix[i, j]
            if vaf > 0.0:
                # Use white text for dark cells, dark text for light cells
                text_color = "white" if vaf > 0.5 else "black"
                ax.text(j, i, f"{vaf:.2f}", ha="center", va="center",
                       color=text_color, fontsize=6, weight="bold")

    # Remove top and right spines
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("VAF", fontsize=10, fontweight="bold")
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}"))

    # ── Title ────────────────────────────────────────────────────────────
    title = f"{alias} — GL_DB Variant VAF Matrix (Clustered)\nShared variants only (≥2 samples)"
    fig.suptitle(
        title,
        fontsize=10, fontweight="bold", y=0.98,
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Variant-sample VAF heatmap saved to: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    run_folder = args.TSOPPI_folder.resolve()
    if not run_folder.is_dir():
        print(f"[ERROR] Not a directory: {run_folder}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir.resolve() if args.output_dir else run_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = run_folder.name

    pcgr_files = discover_pcgr_files(run_folder)
    if not pcgr_files:
        print("[ERROR] No PCGR tiers TSV files found under the given folder.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 1: Per-sample filtering (VAF threshold applied)
    # ------------------------------------------------------------------
    filtered: dict[str, pl.DataFrame] = {}
    for sample_id, tsv_path in pcgr_files.items():
        df = load_and_filter(tsv_path, args.VAF_th)
        filtered[sample_id] = df
        print(
            f"\n{'='*70}\n"
            f"Sample : {sample_id}\n"
            f"Filter : TAL == GL_DB  AND  AF_TUMOR present  AND  AF_TUMOR <= {args.VAF_th}\n"
            f"Hits   : {df.height}\n"
            f"{'='*70}"
        )
        with pl.Config(tbl_rows=-1):
            print(df)

        if args.tmp:
            tmp_path = output_dir / f"{sample_id}_GL_DB_filtered_variants.tsv"
            df.write_csv(tmp_path, separator="\t")
            print(f"  -> Saved: {tmp_path}")

    # ------------------------------------------------------------------
    # Step 2: Complete GL_DB data per sample (no VAF threshold)
    # ------------------------------------------------------------------
    complete = load_complete_gl_db_data(pcgr_files)

    # ------------------------------------------------------------------
    # Step 3: Cross-reference (filtered ref → complete query) and display
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Cross-reference: filtered variants from ref found in complete query data")
    print(f"{'='*70}")
    xref_table = build_cross_reference_table(filtered, complete)
    with pl.Config(tbl_rows=-1):
        print(xref_table)

    xref_path = output_dir / f"{run_name}_GL_DB_cross_reference_table.tsv"
    xref_table.write_csv(xref_path, separator="\t")
    print(f"\nCross-reference table saved to: {xref_path}")

    # ------------------------------------------------------------------
    # Step 4: Intersection count matrix (filtered ref vs complete query)
    # ------------------------------------------------------------------
    matrix = build_intersection_matrix(filtered, complete)
    matrix_path = output_dir / f"{run_name}_GL_DB_intersection_matrix.tsv"
    matrix.write_csv(matrix_path, separator="\t")

    print(f"\n{'='*70}")
    print(f"Intersection matrix  ({len(filtered)} x {len(filtered)} samples)")
    print(f"  Rows (ref)  : variants with AF_TUMOR <= {args.VAF_th}")
    print(f"  Cols (query): all GL_DB variants (no threshold)")
    print(f"Saved to: {matrix_path}")
    print(f"{'='*70}")
    print(matrix)

    # ------------------------------------------------------------------
    # Step 5: Intersection heatmap figure (barplot + heatmap)
    # ------------------------------------------------------------------
    sample_counts = {sid: df.height for sid, df in filtered.items()}
    heatmap_path  = output_dir / f"{run_name}_GL_DB_intersection_heatmap.pdf"
    plot_intersection_heatmap(matrix, sample_counts, heatmap_path, run_name, args.VAF_th, args.alias)

    # ------------------------------------------------------------------
    # Step 6: VAF heatmap — relevant overlaps with actual VAF values
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Building VAF heatmap for relevant overlap variants")
    print(f"{'='*70}")

    variant_ids, sample_ids, vaf_data = build_overlap_vaf_matrix(filtered, complete)
    print(f"Relevant overlap variants: {len(variant_ids)}")

    if len(variant_ids) < 2:
        print(f"[INFO] Found {len(variant_ids)} relevant variant(s); minimum 2 required. Skipping VAF heatmap.")
    else:
        print(f"VAF matrix: {len(variant_ids)} variants x {len(sample_ids)} samples")
        variant_order, sample_order = cluster_vaf_matrix(vaf_data, variant_ids, sample_ids)

        vaf_heatmap_path = output_dir / f"{run_name}_GL_DB_variant_sample_heatmap.pdf"
        plot_variant_sample_heatmap(
            vaf_data, variant_ids, sample_ids,
            variant_order, sample_order,
            vaf_heatmap_path, args.alias,
        )


if __name__ == "__main__":
    main()

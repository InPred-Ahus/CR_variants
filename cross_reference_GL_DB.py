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
_XREF_COLS    = ["AF_TUMOR", "GENOMIC_CHANGE", "SYMBOL", "sampleID_reference", "sampleID_query"]


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


# -------------- #
# File discovery # 
# -------------- #

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


# --------- #
# Filtering # 
# --------- #

def load_gl_db_variants(tsv_path : Path,
                        vaf_th   : float | None = None) -> pl.DataFrame:
    """
    Load GL_DB variants from a PCGR tiers TSV.
    
    Always filters by TAL == "GL_DB" and AF_TUMOR present.
    Optionally applies VAF threshold: if vaf_th is provided, only returns
    variants with AF_TUMOR <= vaf_th.

    Parameters
    ----------
    tsv_path : Path
        Path to the PCGR tiers TSV file
    vaf_th : float | None
        If provided, filter to AF_TUMOR <= vaf_th.
        If None, return all GL_DB variants with AF_TUMOR present.

    Returns
    -------
    DataFrame with selected columns (_VARIANT_COLS)
    """
    df = pl.read_csv(tsv_path, separator="\t", infer_schema_length=0)

    # String-encoded "NA" → null; then cast to float for numeric comparison
    df = df.with_columns(
        pl.col("GLOBAL_AF_1KG").replace("NA", None).cast(pl.Float64),
        pl.col("AF_TUMOR").replace("NA", None).cast(pl.Float64),
    )

    # Base filter: TAL == GL_DB AND AF_TUMOR is present
    df = df.filter(
        (pl.col("TAL") == "GL_DB")
        & pl.col("AF_TUMOR").is_not_null()
    )

    # Optional VAF threshold
    if vaf_th is not None:
        df = df.filter(pl.col("AF_TUMOR") <= vaf_th)

    return df.select(_VARIANT_COLS)


# --------------- #
# Cross-reference # 
# --------------- #

def load_complete_gl_db_data(pcgr_files: dict[str, Path]) -> dict[str, pl.DataFrame]:
    """
    Load complete GL_DB variants from all samples without VAF threshold filtering.
    
    Returns
    -------
    dict[sample_id] → DataFrame with all GL_DB variants (TAL=="GL_DB", AF_TUMOR present)
    """
    complete_data: dict[str, pl.DataFrame] = {}
    
    for sample_id, tsv_path in pcgr_files.items():
        complete_data[sample_id] = load_gl_db_variants(tsv_path, vaf_th=None)
    
    return complete_data


def build_cross_reference_table(filtered : dict[str, pl.DataFrame],
                                complete : dict[str, pl.DataFrame]) -> pl.DataFrame:
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

        # Skip empty reference samples (no filtered variants)
        if ref_df.is_empty():
            continue

        ref_variants = set(ref_df["GENOMIC_CHANGE"].to_list())

        for query_id, query_df in complete.items():

            # Skip self-comparison and empty query samples (no GL_DB variants)
            if query_id == ref_id or query_df.is_empty():
                continue

            # Find shared variants between reference's filtered set and query's complete set
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
                "AF_TUMOR":           pl.Float64,
                "GENOMIC_CHANGE":     pl.Utf8,
                "SYMBOL":             pl.Utf8,
                "sampleID_reference": pl.Utf8,
                "sampleID_query":     pl.Utf8,
            }
        )

    result = pl.concat(rows)

    # Sort by chromosome (numeric) then by full GENOMIC_CHANGE for natural ordering
    result = result.with_columns(
        pl.col("GENOMIC_CHANGE").str.extract(r'^(\d+)', 1).cast(pl.Int64).alias("_chrom_num")
    ).sort(["_chrom_num", "GENOMIC_CHANGE"]).drop("_chrom_num")
    return result


def build_intersection_matrix(filtered: dict[str, pl.DataFrame],
                              complete: dict[str, pl.DataFrame]) -> pl.DataFrame:
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

        # Count how many of query's filtered variants appear in its own complete data
        # This is the diagonal of the matrix (self-count)
        query_variants = set(complete[query_id]["GENOMIC_CHANGE"].to_list()) if not complete[query_id].is_empty() else set()
        
        for ref_id in sample_ids:
            ref_variants = set(filtered[ref_id]["GENOMIC_CHANGE"].to_list()) if not filtered[ref_id].is_empty() else set()
            
            # Count how many ref's filtered variants appear in query's complete data
            # This is the off-diagonal count for (ref, query)
            counts.append(len(ref_variants & query_variants))
        
        data[query_id] = counts

    return pl.DataFrame(data)


def build_overlap_vaf_matrix(filtered: dict[str, pl.DataFrame],
                             complete: dict[str, pl.DataFrame]) -> tuple[list[str], list[str], dict[str, dict[str, float]], set[tuple[str, str]]]:
    """
    Build a VAF matrix restricted to variants that are relevant overlaps:
    variants that passed VAF_th in at least one reference sample AND appear
    in at least one other sample's complete GL_DB data.

    Returns
    -------
    variant_ids    : sorted list of GENOMIC_CHANGE strings
    sample_ids     : sorted list of sample identifiers
    vaf_data       : dict[variant_id][sample_id] = VAF (0.0 if absent in that sample)
    filtered_cells : set of (variant_id, sample_id) tuples that passed VAF_th
    """

    # Collect variants that passed VAF_th in any sample AND appear in another
    # This ensures we only include variants that are relevant for cross-sample contamination analysis.
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

    # Track which cells passed VAF_th (for star annotation in heatmap)
    filtered_cells: set[tuple[str, str]] = set()
    for sample_id, df in filtered.items():
        if df.is_empty():
            continue
        filtered_vars = set(df["GENOMIC_CHANGE"].to_list())
        for variant in filtered_vars:
            if variant in relevant:
                filtered_cells.add((variant, sample_id))

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

    return variant_ids, sample_ids, vaf_data, filtered_cells


def cluster_vaf_matrix(vaf_data    : dict[str, dict[str, float]],
                       variant_ids : list[str],
                       sample_ids  : list[str]) -> tuple[list[int], list[int]]:
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


# ------------- #
# Visualisation #
# ------------- #

def plot_intersection_heatmap(matrix        : pl.DataFrame,
                              sample_counts : dict[str, int],
                              output_path   : Path,
                              vaf_th        : float,
                              alias         : str) -> None:
    """
    Composite figure: barplot of per-sample rare variant counts (top panel) and
    pairwise intersection heatmap (bottom panel).

    Displays the full asymmetric matrix (rows=reference, columns=query) to show
    contamination flow in both directions. Built with matplotlib for vector PDF output.
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

    # Numeric matrix (row i = reference sample i, col j = query sample j)
    # This is asymmetric: mat[ref, query] = ref's filtered variants found in query's complete data
    mat = np.array(
        [
            [matrix.filter(pl.col("sample") == ref).select(qry).item()
             for qry in sample_ids]
            for ref in sample_ids
        ],
        dtype=float,
    )

    # Color scale based on all off-diagonal values (asymmetric contamination signals)
    # Diagonal cells are self-counts (filtered variants per sample)
    off_diag_values = mat[~np.eye(mat.shape[0], dtype=bool)]
    off_diag_values = off_diag_values[off_diag_values > 0]  # exclude zeros
    # Use fixed cap of 50 for good color differentiation; values > 50 saturate to darkest blue
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
    # Show full asymmetric matrix to visualize contamination flow in both directions
    im = ax_heat.imshow(
        mat, cmap=cmap, vmin=0, vmax=vmax,
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
    # Diagonal cells: bold, larger font (self-counts); off-diagonal: standard adaptive text
    for i in range(n):
        for j in range(n):
            is_diagonal = (i == j)
            
            if is_diagonal:
                # Diagonal: bold, larger text for self-counts
                ax_heat.text(
                    j, i, str(int(mat[i, j])),
                    ha="center", va="center",
                    fontsize=10, fontweight="bold", color="#1a1a1a",
                )
            else:
                # Off-diagonal: standard adaptive text for contamination signals
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


def plot_variant_sample_heatmap(vaf_data       : dict[str, dict[str, float]],
                                variant_ids    : list[str],
                                sample_ids     : list[str],
                                variant_order  : list[int],
                                sample_order   : list[int],
                                output_path    : Path,
                                alias          : str,
                                filtered_cells : set[tuple[str, str]] | None = None) -> None:
    """
    Generate a publication-quality VAF heatmap: samples (rows) × variants (columns).
    
    Displays tumor allele frequencies for GL_DB variants across samples with 
    hierarchical clustering (Ward linkage, Euclidean distance) on both axes.
    Restricted to "relevant" variants (passed VAF_th in ≥1 sample AND appear in 
    another sample). Uses ColorBrewer Blues palette (white=0, dark blue=high VAF).
    Cell text: "{vaf:.2f}*" format (star if filtered_cells), adaptive color,
    zero cells masked white. Output: vector PDF (dpi=150).

    Parameters
    ----------
    vaf_data : dict[str, dict[str, float]]
        Nested dictionary mapping variant ID → sample ID → VAF value.
        Structure: vaf_data[variant_id][sample_id] = float (VAF, 0.0-1.0).
        Expected to contain only "relevant" variants (shared across samples);
        missing entries default to 0.0 (variant absent in that sample).
        Variant IDs are GENOMIC_CHANGE strings (e.g., "chr2:g.178422076C>T"),
        sample IDs are from TSOPPI folder structure.

    variant_ids : list[str]
        Sorted list of variant identifiers (GENOMIC_CHANGE format).
        These will be displayed as column headers (x-axis) in the heatmap.
        Expected to match keys in vaf_data.

    sample_ids : list[str]
        Sorted list of TSOPPI sample identifiers.
        These will be displayed as row headers (y-axis) in the heatmap.
        Expected to match keys in nested vaf_data dictionaries.

    variant_order : list[int]
        Leaf indices from hierarchical clustering of variants.
        Integer indices into variant_ids array specifying display order
        (left → right in heatmap). Produced by cluster_vaf_matrix().

    sample_order : list[int]
        Leaf indices from hierarchical clustering of samples.
        Integer indices into sample_ids array specifying display order
        (top → bottom in heatmap). Produced by cluster_vaf_matrix().

    output_path : Path
        Absolute or relative filesystem path for PDF output.
        File will be created or overwritten; parent directory must exist.
        Recommended naming: "{alias}_GL_DB_variant_sample_heatmap.pdf"

    alias : str
        User-provided name for this analysis run.
        Used in plot title and for file naming consistency.
        Example: "TSOPPI_Run_20250915" or "contamination_check_final"

    filtered_cells : set[tuple[str, str]] | None, optional
        Set of (variant_id, sample_id) tuples representing cells that
        passed the VAF threshold in their respective sample.
        Used for star (*) annotation: if (variant, sample) is in this set,
        the cell text will be appended with "*".
        Default: None (no star annotations, treated as empty set).

    Returns
    -------
    None
        Writes a vector PDF file to output_path. Console output includes
        success message with full file path and summary statistics.

    """
    if filtered_cells is None:
        filtered_cells = set()

    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    import numpy as np

    plt.switch_backend("Agg")

    n_variants = len(variant_ids)
    n_samples  = len(sample_ids)

    # Build reordered VAF matrix: samples (rows) × variants (columns)
    reordered_variants = [variant_ids[i] for i in variant_order]
    reordered_samples  = [sample_ids[j]  for j in sample_order]

    vaf_matrix = np.array(
        [[vaf_data[v][s] for v in reordered_variants] for s in reordered_samples],
        dtype=float,
    )

    # Figure geometry — slightly larger cells for a taller heatmap; dot strip stays compact
    cell_in = max(0.5, min(1.2, 12.0 / max(n_variants, 1)))
    dot_h   = 1.0          # compact dot strip
    heat_h  = n_samples * cell_in

    fig_w = max(12.0, n_variants * cell_in + 5.0)   # 5.0 for y-labels + colorbar
    fig_h = max(8.0,  heat_h + dot_h + 3.5)         # 3.5 for title + x-labels

    # constrained_layout respects aspect="equal" in imshow and avoids overlap
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white", constrained_layout=True)
    gs  = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[dot_h, heat_h], hspace=0.0)
    ax_dot  = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1], sharex=ax_dot)

    # ── Dotplot: per-variant VAF distribution across samples ──────────────
    # Each x-position is a variant column (x-axis shared with heatmap).
    # Y-axis is VAF (0 – 1). One dot per sample with VAF > 0.
    # Red dot: (variant, sample) passed VAF_th; dark grey: present but not filtered.
    # Alternating column background: white / light-grey for readability.
    for j in range(n_variants):
        if j % 2 == 1:
            ax_dot.axvspan(j - 0.5, j + 0.5, facecolor="#f0f0f0", zorder=0)

    for j, variant in enumerate(reordered_variants):
        for sample in reordered_samples:
            vaf = vaf_data[variant][sample]
            if vaf > 0.0:
                color = "#721010" if (variant, sample) in filtered_cells else "#767575"
                ax_dot.scatter(j, vaf, color=color, s=35, alpha=0.85,
                               edgecolors="none", zorder=3, clip_on=True)

    for y_ref in (0.0, 0.5, 1.0):
        ax_dot.axhline(y_ref, color="#cccccc", linewidth=0.6, zorder=1)

    ax_dot.set_xlim(-0.5, n_variants - 0.5)
    ax_dot.set_ylim(-0.05, 1.05)
    ax_dot.set_yticks([0.0, 0.5, 1.0])
    ax_dot.set_yticklabels(["0", "0.5", "1"], fontsize=8)
    ax_dot.set_ylabel("VAF", fontsize=9)
    ax_dot.set_xticks(np.arange(n_variants))
    ax_dot.set_xticklabels(reordered_variants, rotation=45, ha="right", fontsize=7)
    ax_dot.tick_params(length=0)
    ax_dot.spines[["top", "right", "bottom"]].set_visible(False)
    ax_dot.spines["left"].set_linewidth(0.8)

    # Legend inside ax_dot (upper-right corner) so it does not displace the colorbar
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#D32F2F",
               markersize=8, label="Passed VAF threshold"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#424242",
               markersize=8, label="Present — not filtered"),
    ]
    ax_dot.legend(handles=legend_elements, loc="upper right",
                  fontsize=8, frameon=False, ncol=1)

    # ── Heatmap — aspect="equal" enforces square cells ────────────────────
    cmap = plt.cm.Blues
    im   = ax_heat.imshow(
        vaf_matrix, cmap=cmap, vmin=0, vmax=1,
        aspect="equal", interpolation="nearest",
    )

    # White rectangles over VAF=0 cells
    for i in range(n_samples):
        for j in range(n_variants):
            if vaf_matrix[i, j] == 0.0:
                ax_heat.add_patch(
                    Rectangle((j - 0.5, i - 0.5), 1, 1,
                              fill=True, facecolor="white", edgecolor="none")
                )

    # Cell text: "{vaf:.2f}" with optional "*" suffix; adaptive color
    for i in range(n_samples):
        for j in range(n_variants):
            vaf = vaf_matrix[i, j]
            if vaf > 0.0:
                text_color = "white" if vaf > 0.5 else "black"
                variant    = reordered_variants[j]
                sample     = reordered_samples[i]
                star       = "*" if (variant, sample) in filtered_cells else ""
                ax_heat.text(j, i, f"{vaf:.2f}{star}", ha="center", va="center",
                             color=text_color, fontsize=6, weight="bold")

    # Axis labels and ticks
    ax_heat.set_xticks(np.arange(n_variants))
    ax_heat.set_yticks(np.arange(n_samples))
    ax_heat.set_xticklabels(reordered_variants, rotation=45, ha="right", fontsize=7)
    ax_heat.set_yticklabels(reordered_samples, fontsize=9)
    ax_heat.set_xlabel("Variant", fontsize=10, fontweight="bold")
    ax_heat.set_ylabel("Sample ID", fontsize=10, fontweight="bold")

    # Minor grid lines for cell delineation
    ax_heat.set_xticks(np.arange(n_variants) - 0.5, minor=True)
    ax_heat.set_yticks(np.arange(n_samples) - 0.5, minor=True)
    ax_heat.grid(which="minor", color="white", linestyle="-", linewidth=0.8)

    ax_heat.spines[["top", "right"]].set_visible(False)
    ax_heat.tick_params(length=0)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04, shrink=0.5)
    cbar.set_label("VAF", fontsize=10, fontweight="bold")
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}"))

    # Title
    title = (
        f"{alias} — GL_DB Variant VAF Matrix\n"
        f"Shared variants only (≥2 samples)   |   * : passed VAF threshold"
    )
    fig.suptitle(title, fontsize=10, fontweight="bold")

    fig.savefig(output_path, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    print(f"; Variant-sample VAF heatmap saved to: {output_path}")


# ----------- #
# Entry point #
# ----------- #


def main() -> None:
    args = parse_args()

    # Validate TSOPPI run folder
    run_folder = args.TSOPPI_folder.resolve()
    if not run_folder.is_dir():
        print(f"[ERROR] Not a directory: {run_folder}", file=sys.stderr)
        sys.exit(1)

    # Prepare output directory
    output_dir = args.output_dir.resolve() if args.output_dir else run_folder
    output_dir.mkdir(parents=True, exist_ok=True)


    pcgr_files = discover_pcgr_files(run_folder)
    if not pcgr_files:
        print("[ERROR] No PCGR tiers TSV files found under the given folder.", file=sys.stderr)
        sys.exit(1)


    # ---------------------------------------------------- #
    # Step 1: Per-sample filtering (VAF threshold applied) #
    # ---------------------------------------------------- #
    filtered: dict[str, pl.DataFrame] = {}
    for sample_id, tsv_path in pcgr_files.items():
        df = load_gl_db_variants(tsv_path, args.VAF_th)
        filtered[sample_id] = df
        print(f"; Sample: {sample_id} - Hits: {df.height}")

        # Export per-sample filtered variants if requested
        if args.tmp:
            tmp_path = output_dir / f"{sample_id}_GL_DB_filtered_variants.tsv"
            df.write_csv(tmp_path, separator="\t")


    # ------------------------------------------------------------------
    # Step 2: Complete GL_DB data per sample (no VAF threshold)
    # ------------------------------------------------------------------
    complete = load_complete_gl_db_data(pcgr_files)


    # ------------------------------------------------------- #
    # Step 3: Cross-reference (filtered ref → complete query) #
    # ------------------------------------------------------- #
    print("; Cross-reference: filtered variants from ref found in complete query data")
    xref_table = build_cross_reference_table(filtered, complete)
    xref_path = output_dir / f"{args.alias}_GL_DB_cross_reference_table.tsv"
    xref_table.write_csv(xref_path, separator="\t")
    print(f"; Cross-reference table: {xref_path}")


    # --------------------------------- #
    # Step 4: Intersection count matrix #
    # --------------------------------- #
    matrix = build_intersection_matrix(filtered, complete)
    matrix_path = output_dir / f"{args.alias}_GL_DB_intersection_matrix.tsv"
    matrix.write_csv(matrix_path, separator="\t")
    print(f"Intersection matrix  ({len(filtered)} x {len(filtered)} samples): {matrix_path}")


    # ------------------------------------------------------- #
    # Step 5: Intersection heatmap figure (barplot + heatmap) #
    # ------------------------------------------------------- #
    sample_counts = {sid: df.height for sid, df in filtered.items()}
    heatmap_path  = output_dir / f"{args.alias}_GL_DB_intersection_heatmap.pdf"
    plot_intersection_heatmap(matrix, sample_counts, heatmap_path, args.VAF_th, args.alias)

    # ------------------------------------------------------------------
    # Step 6: VAF heatmap — relevant overlaps with actual VAF values
    # ------------------------------------------------------------------
    print("Building VAF heatmap for relevant overlap variants")
    variant_ids, sample_ids, vaf_data, filtered_cells = build_overlap_vaf_matrix(filtered, complete)
    print(f"Relevant overlap variants: {len(variant_ids)}")

    if len(variant_ids) < 2:
        print(f"[INFO] Found {len(variant_ids)} relevant variant(s); minimum 2 required. Skipping VAF heatmap.")
    else:
        print(f"VAF matrix: {len(variant_ids)} variants x {len(sample_ids)} samples")
        variant_order, sample_order = cluster_vaf_matrix(vaf_data, variant_ids, sample_ids)

        vaf_heatmap_path = output_dir / f"{args.alias}_GL_DB_variant_sample_heatmap.pdf"
        plot_variant_sample_heatmap(vaf_data, variant_ids, sample_ids,
                                    variant_order, sample_order,
                                    vaf_heatmap_path, args.alias,
                                    filtered_cells)

if __name__ == "__main__":
    main()

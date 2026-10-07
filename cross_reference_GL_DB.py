#!/usr/bin/env python3
"""
cross_reference_GL_DB.py

For each sample in a TSOPPI run folder, filters PCGR variant calls to GL_DB
variants below a given tumor allele frequency (AF_TUMOR) threshold, then cross-
references those filtered germline-database variants across all samples to
produce a pairwise intersection matrix and a combined visualisation.

Variants with no AF_TUMOR value (AF_TUMOR = "NA") are excluded from analysis.

Usage
-----
    uv run cross_reference_GL_DB.py \\
        --TSOPPI_folder PATH        \\
        [--alias NAME]              \\
        [--VAF_th 0.02]             \\
        [--output_dir PATH]         \\
        [--tmp]

Output (all written to output_dir, default: TSOPPI_folder)
----------------------------------------------------------
    {run}_GL_DB_cross_reference_table.tsv   long-format shared-variant table
    {run}_GL_DB_intersection_matrix.tsv     N x N count matrix
    {run}_GL_DB_intersection_heatmap.pdf    combined barplot + heatmap figure
    {sample}_GL_DB_filtered_variants.tsv    per-sample tables (--tmp only)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

# Columns retained throughout the analysis
_VARIANT_COLS = ["TAL", "AF_TUMOR", "GLOBAL_AF_1KG", "GENOMIC_CHANGE", "SYMBOL", "VCF_SAMPLE_ID"]
_XREF_COLS    = ["TAL", "AF_TUMOR", "GLOBAL_AF_1KG", "GENOMIC_CHANGE", "SYMBOL", "sampleID_reference", "sampleID_query"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
        default="CR_GL_DB",
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

def build_cross_reference_table(filtered: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """
    For every (reference, query) pair where reference != query, find variants
    (matched by GENOMIC_CHANGE) from the reference sample that are also present
    in the query sample.

    The resulting long-format table replaces VCF_SAMPLE_ID with two columns:
    sampleID_reference and sampleID_query.  Variant-level data (AF_TUMOR, etc.)
    reflects the query sample's observation.
    """
    rows: list[pl.DataFrame] = []

    for ref_id, ref_df in filtered.items():
        if ref_df.is_empty():
            continue

        ref_variants = set(ref_df["GENOMIC_CHANGE"].to_list())

        for query_id, query_df in filtered.items():
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


def build_intersection_matrix(filtered: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """
    Build a sample × sample integer matrix of GL_DB variant intersection counts.

    Diagonal  : number of filtered GL_DB variants in that sample.
    Off-diagonal: number of variants shared between reference row and query column.
    """
    sample_ids = list(filtered.keys())
    variant_sets: dict[str, set[str]] = {
        sid: set(df["GENOMIC_CHANGE"].to_list())
        for sid, df in filtered.items()
    }

    data: dict[str, list[int]] = {"sample": sample_ids}  # type: ignore[assignment]
    for query_id in sample_ids:
        data[query_id] = [
            len(variant_sets[ref_id] & variant_sets[query_id])
            for ref_id in sample_ids
        ]

    return pl.DataFrame(data)


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
    plt.switch_backend("Agg")      # force non-interactive backend; safe to call after pyplot import
    import matplotlib.gridspec as gridspec
    import matplotlib.colors as mcolors
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

    # White → deep-blue colormap
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "white_to_blue", ["#ffffff", "#1565C0"]
    )
    vmax = mat.max()

    # Figure geometry scales with sample count
    fig_w = max(8.0, n * 2.0)
    fig   = plt.figure(figsize=(fig_w, fig_w * 1.05), facecolor="white")
    # Single-column GridSpec for barplot and heatmap
    gs = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[1, 3],
        hspace=0.04,
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
    # Cell text with adaptive colour (white on dark cells, charcoal on light)
    # Only draw labels for unmasked (upper triangle + diagonal) cells
    for i in range(n):
        for j in range(n):
            if i <= j:  # Upper triangle and diagonal only
                brightness = mat[i, j] / vmax if vmax > 0 else 0
                tc = "white" if brightness > 0.55 else "#1a1a1a"
                ax_heat.text(
                    j, i, str(int(mat[i, j])),
                    ha="center", va="center", fontsize=8.5, color=tc,
                )

    ax_heat.set_xticks(x_pos)
    ax_heat.set_xticklabels(sample_ids, rotation=45, ha="right", fontsize=9)
    ax_heat.set_yticks(x_pos)
    ax_heat.set_yticklabels(sample_ids, fontsize=9)
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
    # Step 1 & 2: per-sample filtering and display
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
    # Step 3 & 4: cross-reference and display
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Cross-reference: shared GL_DB variants across samples")
    print(f"{'='*70}")
    xref_table = build_cross_reference_table(filtered)
    with pl.Config(tbl_rows=-1):
        print(xref_table)

    xref_path = output_dir / f"{run_name}_GL_DB_cross_reference_table.tsv"
    xref_table.write_csv(xref_path, separator="\t")
    print(f"\nCross-reference table saved to: {xref_path}")

    # ------------------------------------------------------------------
    # Step 5: intersection count matrix export
    # ------------------------------------------------------------------
    matrix = build_intersection_matrix(filtered)
    matrix_path = output_dir / f"{run_name}_GL_DB_intersection_matrix.tsv"
    matrix.write_csv(matrix_path, separator="\t")

    print(f"\n{'='*70}")
    print(f"Intersection matrix  ({len(filtered)} x {len(filtered)} samples)")
    print(f"Saved to: {matrix_path}")
    print(f"{'='*70}")
    print(matrix)

    # ------------------------------------------------------------------
    # Step 6: combined barplot + heatmap figure
    # ------------------------------------------------------------------
    sample_counts = {sid: df.height for sid, df in filtered.items()}
    heatmap_path  = output_dir / f"{run_name}_GL_DB_intersection_heatmap.pdf"
    plot_intersection_heatmap(matrix, sample_counts, heatmap_path, run_name, args.VAF_th, args.alias)


if __name__ == "__main__":
    main()

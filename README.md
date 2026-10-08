# CR_variants

Cross-reference rare germline-database (GL_DB) variants across samples in a TSOPPI run to detect cross-sample contamination.

## Algorithm

For each sample, variants in the PCGR tiers TSV are analysed in two passes:

1. **Reference set** — GL_DB variants with AF_TUMOR ≤ VAF_th (the candidate contamination signal: rare in this sample)
2. **Query set** — all GL_DB variants in every other sample, regardless of VAF

The cross-reference looks up the reference set variants in every other sample's query set. If a variant is found, it is reported with its actual VAF in that sample — which may be high, indicating the variant is not rare there. This asymmetric design is intentional: you are asking "where else do my rare variants appear?" not "where do two samples share the same rare variant?"

The intersection matrix counts how many reference-set variants from each sample appear in each other sample's complete GL_DB data.

## Installation

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
cd CR_variants
uv sync
```

Dependencies: `polars`, `matplotlib`, `numpy`, `scipy`

## Container (for offline / HPC environments)

A self-contained Apptainer image embeds all dependencies and requires no installation at runtime.

```bash
# Build (requires internet; do this outside TSD)
apptainer build cr_variants.sif container/cr_variants.def

# Run
apptainer run                    \
    -B /path/to/tsoppi_run:/data \
    cr_variants.sif              \
    --TSOPPI_folder /data        \
    --output_dir /tmp/output
```

Transfer the resulting `cr_variants.sif` (~200 MB) to offline systems via your institution's import portal.

## Usage

```bash
uv run cross_reference_GL_DB.py \
    --TSOPPI_folder data/TSO500_postprocessing/251211_NB552811_0117_AHTJM2BGYX \
    --alias "Cohort_001" \
    --VAF_th 0.02 \
    --output_dir ./results \
    --tmp
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--TSOPPI_folder` | required | TSOPPI run folder with per-sample subdirectories. Directories ending in `_metrics_plots` are skipped. |
| `--alias` | `CR_GL_DB` | Label used in plot titles. |
| `--VAF_th` | `0.02` | AF_TUMOR threshold for the reference set. Variants above this value are not used as candidate signals, but are still retrieved as query data from other samples. |
| `--output_dir` | (TSOPPI_folder) | Output directory, created if absent. |
| `--tmp` | off | Also write per-sample filtered variant tables. |

## Output files

All written to `--output_dir`.

| File | Description |
|------|-------------|
| `{run}_GL_DB_cross_reference_table.tsv` | Long-format table: each row is a (ref_sample, query_sample, variant) match with VAF from the query sample |
| `{run}_GL_DB_intersection_matrix.tsv` | N×N count matrix: row = reference sample, column = query sample, value = number of ref-filtered variants found in query |
| `{run}_GL_DB_intersection_heatmap.pdf` | Barplot (per-sample counts) + intersection heatmap |
| `{run}_GL_DB_variant_sample_heatmap.pdf` | VAF heatmap of overlap variants only, clustered by Ward linkage |
| `{sample}_GL_DB_filtered_variants.tsv` | Per-sample filtered tables (`--tmp` only) |

## Interpretation

**Diagonal** of the intersection matrix: number of GL_DB variants in that sample passing VAF_th.

**Off-diagonal** [row i, col j]: how many of sample i's rare variants were also detected (at any VAF) in sample j. A non-zero value warrants investigation:

- **Contamination** — one sample contains DNA from another
- **Biological relatedness** — family members may share rare variants
- **Systematic error** — check preprocessing and batch effects

Use `_GL_DB_variant_sample_heatmap.pdf` to inspect the actual VAF distribution of overlap variants. A contamination signal typically shows a variant at low VAF in one sample (the donor side) and higher VAF in the contaminated sample.

## Example workflow

```bash
# Run analysis
uv run cross_reference_GL_DB.py \
    --TSOPPI_folder data/TSO500_postprocessing/251211_NB552811_0117_AHTJM2BGYX \
    --alias "Batch_20240101" \
    --output_dir results/
```



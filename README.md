# CR_variants

Cross-reference rare germline-database (GL_DB) variants across samples in a TSOPPI bioinformatics run to identify shared population-rare mutations. This is useful for contamination detection and sample relatedness analysis.

## Overview

`cross_reference_GL_DB.py` filters PCGR variant calls from each sample to identify GL_DB variants with low population frequency (1000 Genomes Project ≤ VAF threshold), then computes a pairwise intersection matrix showing how many rare variants are shared between samples.

**Key features:**
- Filters GL_DB variants by population allele frequency
- Excludes variants with missing frequency data (robust population-based classification)
- Generates per-sample filtered variant tables (optional)
- Produces cross-reference long-format table of shared variants
- Creates intersection count matrix (N × N samples)
- Outputs publication-ready PDF visualization (barplot + heatmap)

## Installation

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/) as the package manager.

```bash
cd CR_variants
uv sync  # Install dependencies from pyproject.toml
```

Dependencies: `polars`, `matplotlib`, `numpy`, `pandas`

## Usage

### Basic example

```bash
uv run cross_reference_GL_DB.py \
    --TSOPPI_folder /path/to/251211_NB552811_0117_AHTJM2BGYX
```

### Full example with all options

```bash
uv run cross_reference_GL_DB.py \
    --TSOPPI_folder /path/to/251211_NB552811_0117_AHTJM2BGYX \
    --VAF_th 0.001 \
    --output_dir ./results \
    --tmp
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--TSOPPI_folder` | PATH | **required** | Path to the TSOPPI run folder containing per-sample subdirectories (e.g., `AU565-D01-C01-X06/`, `BT20-D01-C01-X06/`). Directories ending with `_metrics_plots` are automatically ignored. |
| `--VAF_th` | FLOAT | `0.0005` | Maximum 1000 Genomes Project allele frequency (VAF) threshold for rare variant selection. Only variants with GLOBAL_AF_1KG ≤ this value are retained. Variants with missing frequency data are excluded. |
| `--output_dir` | PATH | (same as TSOPPI_folder) | Directory where all output files are written. Created if absent. Defaults to the input TSOPPI_folder. |
| `--tmp` | flag | off | When set, also exports per-sample filtered variant tables as individual TSV files for further analysis. Useful for inspection and downstream processing. |

## Output Files

All output files are written to `--output_dir` (default: `TSOPPI_folder`).

### 1. **Cross-reference table** (long-format)
**Filename:** `{run_name}_GL_DB_cross_reference_table.tsv`

Columns: `TAL`, `AF_TUMOR`, `GLOBAL_AF_1KG`, `GENOMIC_CHANGE`, `SYMBOL`, `sampleID_reference`, `sampleID_query`

- Lists all variants from each reference sample found in any query sample
- One row per (reference, query, variant) combination
- Useful for identifying specific shared mutations and verifying annotations

### 2. **Intersection matrix** (pairwise counts)
**Filename:** `{run_name}_GL_DB_intersection_matrix.tsv`

Tab-separated table with samples as both rows (reference) and columns (query).

- **Diagonal** (i = j): Total number of rare GL_DB variants in sample i
- **Off-diagonal** (i ≠ j): Number of rare variants from sample i found in sample j

Example:
```
sample                  AU565-D01-C01-X06  BT20-D01-C01-X06  HCC22118-D01-C01-X06  ...
AU565-D01-C01-X06       8                  0                 0
BT20-D01-C01-X06        0                  4                 0
HCC22118-D01-C01-X06    0                  0                 4
```

### 3. **Heatmap visualization**
**Filename:** `{run_name}_GL_DB_intersection_heatmap.pdf`

Publication-ready vector PDF with two panels:

- **Top panel (barplot):** Per-sample count of rare GL_DB variants, labeled with counts
- **Bottom panel (heatmap):** Pairwise intersection counts with:
  - X-axis: Query samples
  - Y-axis: Reference samples
  - Cell values: Number of shared variants (white = 0, dark blue = maximum)
  - Title includes VAF threshold used for filtering

### 4. **Per-sample filtered tables** (optional, with `--tmp`)
**Filename:** `{sample_id}_GL_DB_filtered_variants.tsv`

Columns: `TAL`, `AF_TUMOR`, `GLOBAL_AF_1KG`, `GENOMIC_CHANGE`, `SYMBOL`, `VCF_SAMPLE_ID`

- One file per sample
- Contains all filtered rare GL_DB variants for that sample
- Useful for validation and deeper investigation

## Interpretation

### Diagonal values
Each sample's diagonal value represents its count of rare (GL_DB, population VAF ≤ threshold) germline variants in the PCGR analysis.

### Off-diagonal values
Zero off-diagonal counts indicate no shared rare variants between samples (expected for unrelated individuals).

Non-zero off-diagonal values may indicate:
- **Sample contamination** — one sample contains DNA from another
- **Sample relatedness** — biological relationship (family members may share rare variants)
- **Systematic error** — check data quality and preprocessing

### Interpretation example
If AU565-D01-C01-X06 (8 rare variants) has 0 shared variants with all other samples, it is likely a clean, uncontaminated sample from an unrelated individual.

If BT20-D01-C01-X06 and HCC22118-D01-C01-X06 share rare variants (non-zero off-diagonal), investigate:
1. Experimental design — are they expected to be related?
2. Lab workflow — were they processed together (potential cross-contamination)?
3. Individual variant review — use `_GL_DB_cross_reference_table.tsv` to examine shared mutations

## Example workflow

```bash
# 1. Run analysis with default parameters
uv run cross_reference_GL_DB.py \
    --TSOPPI_folder data/TSO500_postprocessing/251211_NB552811_0117_AHTJM2BGYX

# 2. Review results
cat data/TSO500_postprocessing/251211_NB552811_0117_AHTJM2BGYX/251211_NB552811_0117_AHTJM2BGYX_GL_DB_intersection_matrix.tsv

# 3. If contamination suspected, examine specific variants
grep "AU565-D01-C01-X06.*BT20" \
    data/TSO500_postprocessing/251211_NB552811_0117_AHTJM2BGYX/251211_NB552811_0117_AHTJM2BGYX_GL_DB_cross_reference_table.tsv

# 4. Visualize (open PDF in your viewer)
open data/TSO500_postprocessing/251211_NB552811_0117_AHTJM2BGYX/251211_NB552811_0117_AHTJM2BGYX_GL_DB_intersection_heatmap.pdf
```

## Notes

- **Filter logic:** Rows retained must satisfy: `TAL == "GL_DB"` AND `GLOBAL_AF_1KG is not null` AND `GLOBAL_AF_1KG ≤ VAF_th`
- **Missing frequency data:** Variants with no 1000 Genomes entry (GLOBAL_AF_1KG = "NA") are excluded to ensure population-based rarity classification
- **Performance:** Runtime scales with variant count and sample count; typical runs (5 samples, ~100 rare variants each) complete in seconds
- **File discovery:** Script automatically locates PCGR tiers TSV files using the pattern `{sample_id}_PCGR_*_all/{sample_id}.pcgr_acmg.grch37.snvs_indels.tiers.tsv`

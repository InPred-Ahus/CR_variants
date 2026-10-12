"""
tests/test_cross_reference.py

Unit tests for cross_reference_GL_DB.py.

Test dataset design (VAF_TH = 0.02)
-------------------------------------
Four samples (A, B, C, D) and four variants (V1-V4).
The dataset is intentionally comprehensive: it covers low-VAF variants that
cross into other samples at high VAF (the core contamination signal), variants
above the threshold that must be excluded from the reference set, variants
private to a single sample, and an empty sample as an edge-case guard.

  Variant  | GENOMIC_CHANGE  | A (complete) | B (complete) | C (complete) | D
  V1 chr1  | 1:g.100A>T      | 0.010 *      | 0.600        |  —           | —
  V2 chr2  | 2:g.200G>A      | 0.500        | 0.030        |  —           | —
  V3 chr3  | 3:g.300C>G      | 0.015 *      |  —           | 0.400        | —
  V4 chr12 | 12:g.1200T>C    |  —           | 0.018 *      | 0.800        | —

  * = passed VAF_th=0.02 (included in filtered set)

Derived expectations
--------------------
  Filtered sets:
    A : {V1, V3}   (V2 at 0.50 excluded)
    B : {V4}       (V1 at 0.60, V2 at 0.03 excluded)
    C : {}         (no variant passes 0.02)
    D : {}         (empty sample)

  Cross-reference table (3 rows, sorted chr-numerically):
    V1 → ref=A, query=B, AF_TUMOR=0.60   (low in A, high in B)
    V3 → ref=A, query=C, AF_TUMOR=0.40   (low in A, high in C)
    V4 → ref=B, query=C, AF_TUMOR=0.80   (low in B, high in C)

  Intersection matrix:
         A  B  C  D
    A  [ 2, 1, 1, 0 ]
    B  [ 0, 1, 1, 0 ]
    C  [ 0, 0, 0, 0 ]
    D  [ 0, 0, 0, 0 ]

  Relevant overlap variants (VAF matrix): {V1, V3, V4}
    V2 excluded — above VAF_th in A, never in any filtered set

  VAF values per variant (from complete data):
    V1: A=0.010, B=0.600, C=0.0,   D=0.0
    V3: A=0.015, B=0.0,   C=0.400, D=0.0
    V4: A=0.0,   B=0.018, C=0.800, D=0.0

  Filtered cells (passed VAF_th per sample):
    {(V1, A), (V3, A), (V4, B)}
"""

import sys
from pathlib import Path

import polars as pl
import pytest

# Allow import from the project root (no package install required)
sys.path.insert(0, str(Path(__file__).parent.parent))

from cross_reference_GL_DB import (
    build_cross_reference_table,
    build_intersection_matrix,
    build_overlap_vaf_matrix,
    cluster_vaf_matrix,
)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset construction helpers
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMA = {
    "TAL":            pl.Utf8,
    "AF_TUMOR":       pl.Float64,
    "GENOMIC_CHANGE": pl.Utf8,
    "SYMBOL":         pl.Utf8,
    "VCF_SAMPLE_ID":  pl.Utf8,
}


def _make_df(rows: list[tuple[str, float, str, str]]) -> pl.DataFrame:
    """
    Build a DataFrame matching _VARIANT_COLS.

    Parameters
    ----------
    rows : list of (GENOMIC_CHANGE, AF_TUMOR, SYMBOL, VCF_SAMPLE_ID)
        TAL is always 'GL_DB'.  Pass an empty list for an empty sample.
    """
    if not rows:
        return pl.DataFrame(schema=_SCHEMA)
    return pl.DataFrame({
        "TAL":            ["GL_DB"] * len(rows),
        "AF_TUMOR":       [r[1] for r in rows],
        "GENOMIC_CHANGE": [r[0] for r in rows],
        "SYMBOL":         [r[2] for r in rows],
        "VCF_SAMPLE_ID":  [r[3] for r in rows],
    })


# ─────────────────────────────────────────────────────────────────────────────
# Shared test data (module-level constants — pure, immutable)
# ─────────────────────────────────────────────────────────────────────────────

# All GL_DB variants per sample, regardless of VAF (query set)
COMPLETE: dict[str, pl.DataFrame] = {
    "A": _make_df([
        ("1:g.100A>T",   0.010, "GENE1", "A"),
        ("2:g.200G>A",   0.500, "GENE2", "A"),
        ("3:g.300C>G",   0.015, "GENE3", "A"),
    ]),
    "B": _make_df([
        ("1:g.100A>T",   0.600, "GENE1", "B"),
        ("2:g.200G>A",   0.030, "GENE2", "B"),
        ("12:g.1200T>C", 0.018, "GENE4", "B"),
    ]),
    "C": _make_df([
        ("3:g.300C>G",   0.400, "GENE3", "C"),
        ("12:g.1200T>C", 0.800, "GENE4", "C"),
    ]),
    "D": _make_df([]),  # empty sample
}

# Variants that passed VAF_th=0.02 per sample (reference set)
FILTERED: dict[str, pl.DataFrame] = {
    "A": _make_df([
        ("1:g.100A>T",   0.010, "GENE1", "A"),  # 0.010 ≤ 0.02
        ("3:g.300C>G",   0.015, "GENE3", "A"),  # 0.015 ≤ 0.02
        # V2 (0.50) excluded
    ]),
    "B": _make_df([
        ("12:g.1200T>C", 0.018, "GENE4", "B"),  # 0.018 ≤ 0.02
        # V1 (0.60) and V2 (0.03) excluded
    ]),
    "C": _make_df([]),  # no variants pass threshold
    "D": _make_df([]),  # empty sample
}


# ─────────────────────────────────────────────────────────────────────────────
# Module-scoped fixtures — each function is computed once per test session
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def xref_table() -> pl.DataFrame:
    return build_cross_reference_table(FILTERED, COMPLETE)


@pytest.fixture(scope="module")
def intersection_matrix() -> pl.DataFrame:
    return build_intersection_matrix(FILTERED, COMPLETE)


@pytest.fixture(scope="module")
def overlap_vaf() -> tuple:
    """Returns (variant_ids, sample_ids, vaf_data, filtered_cells)."""
    return build_overlap_vaf_matrix(FILTERED, COMPLETE)


# ─────────────────────────────────────────────────────────────────────────────
# build_cross_reference_table
# ─────────────────────────────────────────────────────────────────────────────

def test_xref_row_count(xref_table: pl.DataFrame) -> None:
    """Three unique (ref, query, variant) combinations are expected."""
    assert xref_table.height == 3


def test_xref_output_columns(xref_table: pl.DataFrame) -> None:
    expected = {"AF_TUMOR", "GENOMIC_CHANGE", "SYMBOL", "sampleID_reference", "sampleID_query"}
    assert set(xref_table.columns) == expected


@pytest.mark.parametrize("variant,ref,query,expected_af", [
    ("1:g.100A>T",   "A", "B", 0.600),  # V1: low in A, found at high VAF in B
    ("3:g.300C>G",   "A", "C", 0.400),  # V3: low in A, found at high VAF in C
    ("12:g.1200T>C", "B", "C", 0.800),  # V4: low in B, found at high VAF in C
])
def test_xref_specific_rows(
    xref_table: pl.DataFrame, variant: str, ref: str, query: str, expected_af: float
) -> None:
    """Each expected cross-reference row must exist with the correct query-side VAF."""
    row = xref_table.filter(
        (pl.col("GENOMIC_CHANGE") == variant)
        & (pl.col("sampleID_reference") == ref)
        & (pl.col("sampleID_query") == query)
    )
    assert row.height == 1, f"Expected exactly 1 row for ({variant}, ref={ref}, query={query})"
    assert row["AF_TUMOR"][0] == pytest.approx(expected_af)


@pytest.mark.parametrize("variant,ref", [
    ("2:g.200G>A", "A"),  # V2 above VAF_th in A — must not appear as a reference entry
])
def test_xref_above_threshold_excluded(
    xref_table: pl.DataFrame, variant: str, ref: str
) -> None:
    """Variants above VAF_th are excluded from the reference set and must not appear as ref."""
    assert xref_table.filter(
        (pl.col("GENOMIC_CHANGE") == variant) & (pl.col("sampleID_reference") == ref)
    ).is_empty()


@pytest.mark.parametrize("ref_sample", ["C", "D"])
def test_xref_empty_reference_produces_no_rows(
    xref_table: pl.DataFrame, ref_sample: str
) -> None:
    """Samples with no filtered variants must contribute no rows as reference."""
    assert xref_table.filter(pl.col("sampleID_reference") == ref_sample).is_empty()


def test_xref_chromosome_sort_order(xref_table: pl.DataFrame) -> None:
    """
    Rows must be chromosome-numerically ordered: chr1 < chr3 < chr12.
    Lexicographic order would place chr12 before chr3; numeric must prevail.
    """
    gc = xref_table["GENOMIC_CHANGE"].to_list()
    first_occurrence = {v: i for i, v in enumerate(gc) if v not in {g for g in gc[:gc.index(v)]} }
    # Build first-appearance order
    seen: dict[str, int] = {}
    for i, v in enumerate(gc):
        if v not in seen:
            seen[v] = i
    assert seen["1:g.100A>T"] < seen["3:g.300C>G"] < seen["12:g.1200T>C"]


# ─────────────────────────────────────────────────────────────────────────────
# build_intersection_matrix
# ─────────────────────────────────────────────────────────────────────────────

def test_intersection_matrix_shape(intersection_matrix: pl.DataFrame) -> None:
    n = len(FILTERED)
    assert intersection_matrix.height == n
    assert len(intersection_matrix.columns) == n + 1  # +1 for "sample" column


@pytest.mark.parametrize("ref,query,expected", [
    # Diagonal (self-overlap): ref's filtered variants in its own complete data
    ("A", "A", 2),  # A filtered={V1,V3}; both in complete_A
    ("B", "B", 1),  # B filtered={V4};    V4 in complete_B
    ("C", "C", 0),  # C has no filtered variants
    ("D", "D", 0),  # D is empty
    # Off-diagonal: ref's filtered variants found in query's complete data
    ("A", "B", 1),  # V1 shared between filtered_A and complete_B
    ("A", "C", 1),  # V3 shared between filtered_A and complete_C
    ("A", "D", 0),  # D is empty — no variants
    ("B", "A", 0),  # V4 not present in complete_A
    ("B", "C", 1),  # V4 shared between filtered_B and complete_C
    ("B", "D", 0),  # D is empty
    ("C", "A", 0),  # C has no filtered variants
    ("C", "B", 0),  # C has no filtered variants
    ("D", "B", 0),  # D has no filtered variants
])
def test_intersection_matrix_cell(
    intersection_matrix: pl.DataFrame, ref: str, query: str, expected: int
) -> None:
    val = intersection_matrix.filter(pl.col("sample") == ref).select(query).item()
    assert val == expected, f"matrix[{ref},{query}]: expected {expected}, got {val}"


# ─────────────────────────────────────────────────────────────────────────────
# build_overlap_vaf_matrix
# ─────────────────────────────────────────────────────────────────────────────

def test_overlap_relevant_variant_set(overlap_vaf: tuple) -> None:
    """Only variants that are cross-sample (filtered in ≥1, present in another) are included."""
    variant_ids, _, _, _ = overlap_vaf
    assert set(variant_ids) == {"1:g.100A>T", "3:g.300C>G", "12:g.1200T>C"}


def test_overlap_v2_excluded(overlap_vaf: tuple) -> None:
    """V2 is above VAF_th in the only sample that has it (A); it must be absent."""
    variant_ids, _, _, _ = overlap_vaf
    assert "2:g.200G>A" not in variant_ids


def test_overlap_all_samples_present(overlap_vaf: tuple) -> None:
    _, sample_ids, _, _ = overlap_vaf
    assert set(sample_ids) == set(COMPLETE.keys())


@pytest.mark.parametrize("variant,sample,expected_vaf", [
    # V1: low in A (contamination source), high in B (contamination target)
    ("1:g.100A>T",   "A", 0.010),
    ("1:g.100A>T",   "B", 0.600),
    ("1:g.100A>T",   "C", 0.0),    # V1 absent from C
    # V3: low in A, high in C
    ("3:g.300C>G",   "A", 0.015),
    ("3:g.300C>G",   "C", 0.400),
    ("3:g.300C>G",   "B", 0.0),    # V3 absent from B
    # V4: low in B, high in C
    ("12:g.1200T>C", "B", 0.018),
    ("12:g.1200T>C", "C", 0.800),
    ("12:g.1200T>C", "A", 0.0),    # V4 absent from A
])
def test_overlap_vaf_values(
    overlap_vaf: tuple, variant: str, sample: str, expected_vaf: float
) -> None:
    _, _, vaf_data, _ = overlap_vaf
    assert vaf_data[variant][sample] == pytest.approx(expected_vaf)


@pytest.mark.parametrize("variant,sample,expected", [
    ("1:g.100A>T",   "A", True),   # V1 passed VAF_th in A
    ("3:g.300C>G",   "A", True),   # V3 passed VAF_th in A
    ("12:g.1200T>C", "B", True),   # V4 passed VAF_th in B
    ("1:g.100A>T",   "B", False),  # V1 present in B at 0.60, did NOT pass threshold
    ("3:g.300C>G",   "C", False),  # V3 in C complete but C has no filtered variants
    ("12:g.1200T>C", "A", False),  # V4 absent from A entirely
    ("12:g.1200T>C", "C", False),  # V4 in C complete but C has no filtered variants
])
def test_overlap_filtered_cells(
    overlap_vaf: tuple, variant: str, sample: str, expected: bool
) -> None:
    _, _, _, filtered_cells = overlap_vaf
    assert ((variant, sample) in filtered_cells) == expected


# ─────────────────────────────────────────────────────────────────────────────
# cluster_vaf_matrix
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n_variants,n_samples", [
    (3, 4),   # standard case matching the real dataset
    (2, 2),   # minimum size for which clustering is possible
    (1, 4),   # single variant: dendrogram falls back to [0]
    (3, 1),   # single sample: dendrogram falls back to [0]
    (5, 10),  # larger matrix for robustness
])
def test_cluster_returns_valid_permutation(n_variants: int, n_samples: int) -> None:
    """
    cluster_vaf_matrix must return permutations (sorted == range) of the
    correct length for any valid matrix dimension.
    """
    variants = [f"V{i}" for i in range(n_variants)]
    samples  = [f"S{i}" for i in range(n_samples)]
    # Deterministic but varied values to avoid degenerate all-zero matrices
    vaf_data = {
        v: {s: ((i * 7 + j * 3) % 100) / 100.0
            for j, s in enumerate(samples)}
        for i, v in enumerate(variants)
    }

    variant_order, sample_order = cluster_vaf_matrix(vaf_data, variants, samples)

    assert sorted(variant_order) == list(range(n_variants))
    assert sorted(sample_order)  == list(range(n_samples))


def test_cluster_with_real_dataset(overlap_vaf: tuple) -> None:
    """Clustering on the full test dataset returns valid permutations."""
    variant_ids, sample_ids, vaf_data, _ = overlap_vaf
    variant_order, sample_order = cluster_vaf_matrix(vaf_data, variant_ids, sample_ids)

    assert sorted(variant_order) == list(range(len(variant_ids)))
    assert sorted(sample_order)  == list(range(len(sample_ids)))

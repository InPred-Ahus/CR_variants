#!/usr/bin/env bash
# VAF threshold parameter sweep for cross_reference_GL_DB.py
# Tests variant cross-reference analysis across multiple VAF thresholds

set -e  # Exit on any error

TSOPPI_FOLDER="data/TSO500_postprocessing/251211_NB552811_0117_AHTJM2BGYX"
VAF_THRESHOLDS=(0.02 0.05 0.15 0.2 0.25)

echo "Starting VAF threshold sweep..."
echo

for VAF_TH in "${VAF_THRESHOLDS[@]}"; do
    OUTPUT_DIR="TEST/xref_variants_VAF_${VAF_TH}"
    ALIAS="Test_VAF_${VAF_TH}"
    
    echo "Running with VAF_th = $VAF_TH"
    mkdir -p "$OUTPUT_DIR"
    
    uv run cross_reference_GL_DB.py      \
        --TSOPPI_folder "$TSOPPI_FOLDER" \
        --alias "$ALIAS"                 \
        --output_dir "$OUTPUT_DIR"       \
        --VAF_th "$VAF_TH"               ;
    
    echo "Completed: $OUTPUT_DIR"
    echo
done
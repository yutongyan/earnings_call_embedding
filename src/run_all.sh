#!/bin/bash
# Master script to run the full PEAD.txt replication pipeline
set -e

echo "=== Step 1: Parse XML transcripts ==="
python3 src/parse_transcripts.py \
    --input /scratch/homo_silicus_ceo/data/streetevent_transcripts/archive \
    --output data/transcripts_parsed.parquet \
    --start_year 2008 --end_year 2019

echo ""
echo "=== Step 2: Build merged dataset ==="
echo "(Using existing WRDS data from /scratch/llm_in_asset_pricing/wrds_raw/)"
python3 src/build_dataset.py \
    --transcripts data/transcripts_parsed.parquet \
    --output data/merged_dataset.parquet

echo ""
echo "=== Step 3: Run rolling-window regressions ==="
python3 src/run_pead_regression.py \
    --input data/merged_dataset.parquet \
    --output data/sue_txt_results.parquet

echo ""
echo "=== Pipeline complete ==="

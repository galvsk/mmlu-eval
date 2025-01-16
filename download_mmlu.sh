#!/bin/bash

# Create data directory structure
mkdir -p data/raw_mmlu
cd data/raw_mmlu

# Use git clone with depth 1 and sparse checkout in one command
git clone --depth 1 --filter=blob:none --sparse https://huggingface.co/datasets/cais/mmlu .

# Only checkout the "all" folder
git sparse-checkout set all

# Fetch LFS objects only for the "all" folder
git lfs pull --include="all/*"

echo "Download completed! Raw MMLU files are in data/raw_mmlu"
echo "Run generate_dataframes.py to process into data/ref_dataframes"

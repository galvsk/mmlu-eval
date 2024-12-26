#!/bin/bash

# Create a directory for the download
mkdir -p MMLU
cd MMLU

# Use git clone with depth 1 and sparse checkout in one command
git clone --depth 1 --filter=blob:none --sparse https://huggingface.co/datasets/cais/mmlu .

# Only checkout the "all" folder
git sparse-checkout set all

# Fetch LFS objects only for the "all" folder
git lfs pull --include="all/*"

echo "Download completed! Files are in the MMLU directory"

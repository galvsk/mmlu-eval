# Claude MMLU Experimenter

A toolkit for running and analyzing MMLU (Massive Multitask Language Understanding) benchmark experiments using Anthropic's Claude API. This project provides a structured way to download MMLU and run systematic experiments with Claude models.

## Features

- Automated MMLU dataset download and processing
- Structured experiment management with checkpointing
- Progress tracking and result analysis
- Configurable prompting strategies
- Support for multiple Claude models
- Resumable experiments

## Prerequisites

- Python 3.8+
- Anthropic API key
- Git LFS (for downloading MMLU dataset)
- macOS for the default API key management (can be modified for other platforms)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/claude-mmlu-experimenter.git
cd claude-mmlu-experimenter
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the MMLU dataset:
```bash
./download_mmlu.sh
```

## Setup

1. Set up your Anthropic API key:
   - The default implementation expects the API key to be stored in the macOS keychain
   - Modify `utils.py` if you're using a different operating system or key storage method

2. Generate the formatted datasets:
```bash
python generate_dataframes.py
```

This will create processed parquet files in the `ref_dataframes` directory.

## Running Experiments

### Quick Start

To run a basic experiment:

```bash
python run_experiment.py \
    --exp-path experiments/baseline \
    --df-path ref_dataframes/mmlu_test.parquet \
    --desc "Baseline test run" \
    --max-questions 100
```

### Command Line Arguments

- `--exp-path`: Directory to store experiment results
- `--df-path`: Path to MMLU dataset parquet file
- `--desc`: Description of the experiment (required for new experiments)
- `--max-questions`: Maximum number of questions to process (optional)
- `--save-frequency`: How often to save results in number of questions (default: 10)

### Testing the Setup

You can test your setup with a single question:

```bash
python test_claude_api.py
```

## Project Structure

- `download_mmlu.sh`: Script to download MMLU dataset
- `generate_dataframes.py`: Processes raw MMLU data into parquet format
- `mmlu_experimenter.py`: Core experiment management class
- `run_experiment.py`: CLI for running experiments
- `test_claude_api.py`: Simple test script for API setup
- `utils.py`: Utility functions and prompt templates

## Implementation Details

### Experiment Management

The `MMLUExperimenter` class handles:
- Experiment initialization and resumption
- Progress tracking
- Checkpointing
- Result analysis
- Error handling and retries

### Data Processing

The `MMLUData` class provides:
- Dataset loading and validation
- Data formatting
- Summary statistics
- Sample question generation

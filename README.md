# MMLU Experimentation Framework

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

This repository contains a framework for running experiments with Large Language Models (LLMs) on the Massive Multitask Language Understanding (MMLU) benchmark. It currently supports Claude and Deepseek models, with different prompt formatting strategies.

## Features

- Support for multiple LLM APIs:
  - Claude (via Anthropic API)
  - Deepseek (via OpenAI-compatible API)
- Various prompt formatting strategies:
  - Default: Standard multiple-choice format
  - Permuted: Randomly shuffles answer choices
  - Uppercase: Converts entire prompt to uppercase
  - Random Case: Randomly varies character case
  - Duplicate Wrong: Duplicates incorrect choices (7 total options)
- Experiment management:
  - Automatic progress tracking and checkpointing
  - Results analysis and accuracy calculation
  - Resume capability for interrupted experiments

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up API keys:
- Store your API keys securely using the macOS Keychain:
  - Claude API key with service name: `claude-api-key`
  - Deepseek API key with service name: `deepseek-api-key`
  - Include the correct email for your associated api keys

## Dataset Preparation

1. Download the MMLU dataset:
```bash
./download_mmlu.sh
```

2. Generate formatted dataframes:
```bash
python generate_dataframes.py
```

This will create parquet files in the `ref_dataframes` directory:
- `mmlu_train.parquet`: Training dataset
- `mmlu_test.parquet`: Test dataset

## Running Experiments

Use the `run_experiment.py` script to conduct experiments. Example usage:

```bash
python run_experiment.py \
    --exp-path experiments/baseline \
    --df-path ref_dataframes/mmlu_test.parquet \
    --desc "Baseline test run" \
    --max-questions 100 \
    --api claude \
    --prompt-style permuted
```

### Command Line Arguments

- `--exp-path`: Directory to store experiment results
- `--df-path`: Path to MMLU dataset parquet file
- `--desc`: Description of the experiment (required for new experiments)
- `--max-questions`: Maximum number of questions to process
- `--api`: Which model API to use (`claude` or `deepseek`)
- `--save-frequency`: How often to save results (default: 10 questions)
- `--prompt-style`: Prompt formatting style to use:
  - `default`: Standard format
  - `permuted`: Randomly shuffled choices
  - `uppercase`: All uppercase text
  - `randomcase`: Random case variations
  - `duplicatewrong`: Duplicates incorrect choices

## Testing

To test individual components:

```bash
# Test API connectivity
python test_api.py

# Test prompt formatters
python mmlu_formatter.py
```

## Project Structure

- `run_experiment.py`: Main script for running experiments
- `mmlu_experimenter.py`: Core experimentation logic
- `mmlu_formatter.py`: Different prompt formatting strategies
- `model_api.py`: API interfaces for different LLMs
- `utils.py`: Utility functions
- `generate_dataframes.py`: Dataset preparation
- `test_api.py`: API testing utilities

## Experiment Results

Results are stored in the experiment directory specified by `--exp-path`:
- `config.json`: Experiment configuration and metadata
- `results.parquet`: Detailed results including predictions and accuracies

## Contributing

Feel free to open issues or submit pull requests with improvements. Some areas for potential enhancement:
- Additional LLM API support
- New prompt formatting strategies
- Extended results analysis
- Batch processing capabilities

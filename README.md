# MMLU Evaluation Framework

## Overview

This project provides a comprehensive framework for evaluating language models on the Massive Multitask Language Understanding (MMLU) benchmark. The framework supports:

- Multiple model APIs (currently Claude and Deepseek)
- Flexible prompt formatting strategies
- Detailed performance analysis
- LLM generated incorrect answers

## Features

- Multiple prompt formatting styles:
  - Default formatting : standard prompt (see `mmlu_eval/formatter.py`)
  - Permuted choice order : randomly swap order of the answers
  - Uppercase formatting : change all alphabetical characters to uppercase
  - Random case formatting : randomly alternative between uppercase and lowercase alphabetical characters
  - Duplicate wrong answers : repeat all incorrect answers and permute order (six wrong, one right)

- Robust experiment tracking
  - Checkpointing
  - Progress saving
  - Detailed result logging

- Alternative answer generation
  - Generate challenging incorrect answers
  - Support for different difficulty levels

## Prerequisites

- Python 3.8+
- API keys for Claude and/or Deepseek
- Mac with Keychain access (for API key management)

### API Key Management

The project uses Mac's system keychain to securely store API keys. 

1. Set up API keys in the Keychain:
   - For Claude API key:
     ```bash
     security add-generic-password -a "your_email@example.com" -s "claude-api-key" -w "YOUR_CLAUDE_API_KEY"
     ```
   - For Deepseek API key:
     ```bash
     security add-generic-password -a "your_email@example.com" -s "deepseek-api-key" -w "YOUR_DEEPSEEK_API_KEY"
     ```

2. Key retrieval is handled by `mmlu_eval/utils.py`:
   - The `get_api_key()` function automatically retrieves keys from the system keychain
   - Supports both Claude and Deepseek API key management

Note: Replace `your_email@example.com` with the email associated with your API keys, and use your actual API keys.

### Installation

1. Clone the repository
```bash
git clone https://github.com/galvsk/mmlu-eval.git
cd mmlu-eval
```

2. Install the package in editable mode
```bash
pip install -e .
```

## Quick Start

### 1. Preparing Data

1. Download MMLU dataset
```bash
sh download_mmlu.sh
```

2. Generate reference dataframes
```bash
python scripts/generate_dataframes.py
```

This two-step process:
- Downloads the raw MMLU dataset using the provided shell script
- Processes the raw data and generates reference dataframes for evaluation

### 2. Running MMLU evaluation experiments

```bash
python scripts/run_mmlu_eval.py \
    --experiment baseline \
    --desc "Baseline test run" \
    --max-questions 100 \
    --api claude \
    --prompt-style default
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

### 3. Generating new answer choices from LLM APIs

The alternative answer generation is designed to systematically create challenging yet incorrect responses from language models. This helps in:
- Analyzing model vulnerability to plausible but incorrect options
- Understanding how different language models generate incorrect answers
- Probing the reasoning capabilities of AI models

Key features of alternative answer generation:
- Generates multiple incorrect answers for each question
- Configurable difficulty of incorrect responses
- Supports generation from different model APIs (Claude, Deepseek)
- Creates datasets for further analysis of model performance


```bash
python scripts/run_answer_generator.py \
    --df-path ref_dataframes/mmlu_test.parquet \
    --api claude
    --seed 123
```

### Command Line Arguments

- `--df-path`: Path to MMLU dataset parquet file
- `--api`: Which model API to use (`claude` or `deepseek`)
- `--seed`: Random seed to ensure reproducibility of randomly subsampled dataframe

## Acknowledgments

- MMLU Benchmark Creators
- BlueDot Impact for API compute credits
- Claude 3.5 Sonnet for help in developing codebase
- Anthropic and Deepseek for API access

MIT License

Copyright (c) 2024 Galvin Khara

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

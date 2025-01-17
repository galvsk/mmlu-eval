# MMLU Evaluation Framework

## Overview

This project provides a comprehensive framework for evaluating language models on the Massive Multitask Language Understanding (MMLU) benchmark. The framework supports:

- Multiple model APIs (currently Claude and Deepseek)
- Flexible prompt formatting strategies
- LLM-generated alternative answers
- Detailed performance analysis

## Features

### Evaluation Modes

1. Standard MMLU Evaluation
   - Default formatting: standard four-choice prompts
   - Permuted choice order: randomly swap order of answers
   - Uppercase formatting: convert all text to uppercase
   - Random case formatting: randomly alternate between upper and lowercase
   - Duplicate wrong answers: six wrong answers, one right (with permutation)

2. Alternative Answer Evaluation
   - Evaluate models using LLM-generated incorrect answers
   - Two modes:
     - `generated_only`: One correct + three generated wrong answers
     - `all_answers`: One correct + three original + three generated wrong answers
   - Automatic answer permutation for each question

### Framework Features
- Robust experiment tracking
  - Checkpointing
  - Progress saving
  - Detailed result logging
- Support for multiple model APIs
- Configurable evaluation parameters

## Prerequisites

- Python 3.8+
- API keys for Claude and/or Deepseek
- Mac with Keychain access (for API key management)

### API Key Management

The project uses Mac's system keychain to securely store API keys.

1. Set up API keys in the Keychain:
   ```bash
   # For Claude API key
   security add-generic-password -a "your_email@example.com" -s "claude-api-key" -w "YOUR_CLAUDE_API_KEY"
   
   # For Deepseek API key
   security add-generic-password -a "your_email@example.com" -s "deepseek-api-key" -w "YOUR_DEEPSEEK_API_KEY"
   ```

2. Key retrieval is handled automatically by the framework

Note: Replace `your_email@example.com` with your API-associated email.

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

## Usage Guide

### 1. Data Preparation

1. Download MMLU dataset
```bash
sh download_mmlu.sh
```

2. Generate reference dataframes
```bash
python scripts/generate_dataframes.py
```

3. (Optional) Generate alternative answers
```bash
python scripts/run_answer_generator.py \
    --df-path ref_dataframes/mmlu_test.parquet \
    --api claude
```

### 2. Running Evaluations

The framework supports both standard MMLU evaluation and alternative answer evaluation through a single interface:

#### Standard MMLU Evaluation
```bash
python scripts/run_mmlu_eval.py \
    --experiment baseline \
    --desc "Standard MMLU evaluation" \
    --max-questions 100 \
    --api claude \
    --prompt-style permuted
```

#### Alternative Answer Evaluation
```bash
python scripts/run_mmlu_eval.py \
    --experiment alt_answers \
    --desc "Testing with generated answers" \
    --df-path data/generated_dataframes/claude_generated_dataframe.parquet \
    --api claude \
    --alternative-mode generated_only  # or 'all_answers'
```

### Command Line Arguments

Common Arguments:
- `--experiment`: Name of experiment (required)
- `--desc`: Description of the experiment (required)
- `--df-path`: Path to dataset parquet file
- `--max-questions`: Maximum number of questions to process
- `--api`: Which model API to use (`claude` or `deepseek`)
- `--save-frequency`: How often to save results (default: 100 questions)

Standard MMLU Arguments:
- `--prompt-style`: Formatting style (`default`, `permuted`, `uppercase`, `randomcase`, `duplicatewrong`)

Alternative Evaluation Arguments:
- `--alternative-mode`: Evaluation mode (`generated_only` or `all_answers`)
  - `generated_only`: Uses one correct + three generated wrong answers
  - `all_answers`: Uses one correct + three original + three generated wrong answers

Note: When running alternative evaluation, `--prompt-style` is ignored as the formatter is determined by the evaluation mode.

## Acknowledgments

- MMLU Benchmark Creators
- BlueDot Impact for API compute credits
- Claude 3.5 Sonnet for help in developing codebase
- Anthropic and Deepseek for API access

## License

MIT License

Copyright (c) 2024 Galvin Khara

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

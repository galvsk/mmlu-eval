from pathlib import Path


# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Define common data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_MMLU_DIR = DATA_DIR / "raw_mmlu"
REF_DATA_DIR = DATA_DIR / "ref_dataframes"
ALTERNATIVE_DATA_DIR = DATA_DIR / "generated_dataframes"
CLAUDE_LOGS_DIR = DATA_DIR / "claude_logs"
DEEPSEEK_LOGS_DIR = DATA_DIR / "deepseek_logs"

def get_experiment_path(name: str, api: str) -> Path:
    """Get path to an experiment directory based on API type"""
    if api == 'claude':
        return CLAUDE_LOGS_DIR / name
    elif api == 'deepseek':
        return DEEPSEEK_LOGS_DIR / name
    else:
        raise ValueError(f"Unsupported API type: {api}")

# Common MMLU dataframe files
MMLU_TRAIN_FILE = REF_DATA_DIR / "mmlu_train.parquet"
MMLU_TEST_FILE = REF_DATA_DIR / "mmlu_test.parquet"

# %%
import re
import numpy as np
import pandas as pd


# %%
claude_trn = pd.read_parquet('../claude_logs/baseline_train/results.parquet')
deepseek_trn = pd.read_parquet('../deepseek_logs/baseline_train/results.parquet')

# %%
# Deepseek was significantly worse at following the instruction to only output A-D
# Thus we saved responses, and can do some straightforward formatting

def format_deepseek_responses(response: str) -> str:
    """
    Extracts valid MMLU response (A-D) if present at start of string.
    Returns 'INVALID' if no valid response found.
    """
    if pd.isna(response):
        return 'INVALID'
    if response in ['A', 'B', 'C', 'D']:
        return response
       
    match = re.match(r'^[^A-D]*([A-D])', response, re.IGNORECASE)
    return match.group(1).upper() if match else 'INVALID'

deepseek_trn['classified'] = deepseek_trn['response'].apply(format_deepseek_responses)

# %%
print(deepseek_trn.predicted.value_counts(dropna=False))
print(deepseek_trn.response.value_counts(dropna=False))
print(deepseek_trn.classified.value_counts(dropna=False))
# %%

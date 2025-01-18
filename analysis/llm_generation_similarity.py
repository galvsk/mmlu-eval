# %%
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from mmlu_eval.paths import DATA_DIR


# %%
def get_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts."""
    text1, text2 = text1.lower(), text2.lower()
    
    # Word-based Jaccard
    words1 = set(text1.split())
    words2 = set(text2.split())
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    jaccard = intersection / union if union > 0 else 0
    
    # Sequence similarity
    sequence_sim = SequenceMatcher(None, text1, text2).ratio()
    
    return max(jaccard, sequence_sim)

def get_max_similarity_for_row(source_wrongs: np.ndarray, reference_wrongs: np.ndarray) -> np.ndarray:
    """For each source wrong answer, get max similarity with any reference wrong answer."""
    return np.array([
        max(get_text_similarity(src, ref) for ref in reference_wrongs)
        for src in source_wrongs
    ])

def analyze_model_vs_original(df_model: pd.DataFrame) -> pd.DataFrame:
    """Analyze similarities between model's generated wrongs and original wrongs."""
    df_model['similarity_to_original'] = df_model.apply(
        lambda row: get_max_similarity_for_row(
            row['generated_wrong_answers'], 
            row['original_wrong_answers']
        ),
        axis=1
    )
    df_model['mean_similarity_to_original'] = df_model['similarity_to_original'].apply(np.mean)
    return df_model

def analyze_model_vs_model(df_claude: pd.DataFrame, df_deepseek: pd.DataFrame) -> pd.DataFrame:
    """Analyze similarities between Claude's and DeepSeek's generated wrongs."""
    # Ensure dataframes are aligned
    assert len(df_claude) == len(df_deepseek), "Dataframes must have same length"
    
    # Create new dataframe for model comparison
    df_comparison = pd.DataFrame()
    
    # Compare generated wrongs between models using loc/iloc accessor
    df_comparison['model_vs_model'] = [
        get_max_similarity_for_row(
            df_claude.loc[i, 'generated_wrong_answers'], 
            df_deepseek.loc[i, 'generated_wrong_answers']
        )
        for i in range(len(df_claude))
    ]
    
    df_comparison['mean_model_similarity'] = df_comparison['model_vs_model'].apply(np.mean)
    return df_comparison

def get_overall_stats(df_claude: pd.DataFrame, df_deepseek: pd.DataFrame, df_comparison: pd.DataFrame) -> dict:
    """Calculate overall statistics for all comparisons."""
    return {
        'claude_vs_original': {
            'mean': df_claude['mean_similarity_to_original'].mean(),
            'std': df_claude['mean_similarity_to_original'].std(),
            'median': df_claude['mean_similarity_to_original'].median(),
            'q1': df_claude['mean_similarity_to_original'].quantile(0.25),
            'q3': df_claude['mean_similarity_to_original'].quantile(0.75)
        },
        'deepseek_vs_original': {
            'mean': df_deepseek['mean_similarity_to_original'].mean(),
            'std': df_deepseek['mean_similarity_to_original'].std(),
            'median': df_deepseek['mean_similarity_to_original'].median(),
            'q1': df_deepseek['mean_similarity_to_original'].quantile(0.25),
            'q3': df_deepseek['mean_similarity_to_original'].quantile(0.75)
        },
        'claude_vs_deepseek': {
            'mean': df_comparison['mean_model_similarity'].mean(),
            'std': df_comparison['mean_model_similarity'].std(),
            'median': df_comparison['mean_model_similarity'].median(),
            'q1': df_comparison['mean_model_similarity'].quantile(0.25),
            'q3': df_comparison['mean_model_similarity'].quantile(0.75)
        }
    }

# %%
claude_generations = pd.read_parquet(f"{DATA_DIR}/generated_dataframes/claude_generated_dataframe.parquet")
deepseek_generations = pd.read_parquet(f"{DATA_DIR}/generated_dataframes/deepseek_generated_dataframe.parquet")

# %%
# Analyze each model vs original
df_claude = analyze_model_vs_original(claude_generations)
df_deepseek = analyze_model_vs_original(deepseek_generations)

# %%
# Analyze models against each other
df_comparison = analyze_model_vs_model(df_claude, df_deepseek)

# %%
# Get overall statistics
stats = get_overall_stats(df_claude, df_deepseek, df_comparison)

# %%
print(stats)
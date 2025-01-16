# %%
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from utils import bootstrap_test

# %% Load in dataframes for each prompt format
claude_exps = {'Baseline': '../claude_logs/baseline_test/results.parquet',
               'Permuted': '../claude_logs/permuted_test/results.parquet',
               'Uppercase': '../claude_logs/uppercase_test/results.parquet',
               'Random case': '../claude_logs/randomcase_test/results.parquet',
               'Duplicate wrongs': '../claude_logs/duplicate_permuted_wrongs_test/results.parquet'
}

claude_dfs = {}
for exp, fpath in claude_exps.items():
    df = pd.read_parquet(fpath)
    df = df[['answer', 'predicted']]
    claude_dfs[exp] = df

deepseek_exps = {'Baseline': '../deepseek_logs/baseline_test/results.parquet',
                 'Permuted': '../deepseek_logs/permuted_test/results.parquet',
                 'Uppercase': '../deepseek_logs/uppercase_test/results.parquet',
                 'Random case': '../deepseek_logs/randomcase_test/results.parquet',
                 'Duplicate wrongs': '../deepseek_logs/duplicate_permuted_wrongs_test/results.parquet'
}

deepseek_dfs = {}
for exp, fpath in deepseek_exps.items():
    df = pd.read_parquet(fpath)
    df = df[['answer', 'predicted']]
    deepseek_dfs[exp] = df

# %% Get bootstrap results for each experiment
claude_results = {}
for exp, df in claude_dfs.items():
    results = bootstrap_test(df)
    claude_results[exp] = results

# %%
deepseek_results = {}
for exp, df in deepseek_dfs.items():
    results = bootstrap_test(df)
    deepseek_results[exp] = results

# %%
def plot_experiment_performance(results, model_name, figsize=(10, 6)):
    """
    Plot performance across different experimental conditions for a single model.
    
    Parameters:
    -----------
    results : dict
        Dictionary of results with structure:
        {condition: {'mean': float, 'ci_lower': float, 'ci_upper': float}}
    model_name : str
        Name of the model for the plot title
    """
    plt.style.use('seaborn-v0_8-colorblind')
    
    # Define order of conditions
    condition_order = ['Baseline', 'Permuted', 'Uppercase', 'Random case', 'Duplicate wrongs']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up positions for bars
    x = np.arange(len(condition_order))
    width = 0.5  # Slightly thinner bars since we have fewer categories
    
    # Plot bars and error bars
    bars = ax.bar(x, 
                 [results[condition]['mean'] for condition in condition_order],
                 width,
                 yerr=[[results[condition]['mean'] - results[condition]['ci_lower'] for condition in condition_order],
                       [results[condition]['ci_upper'] - results[condition]['mean'] for condition in condition_order]],
                 capsize=5)
    
    # Add text labels with mean and CIs
    for idx, rect in enumerate(bars):
        height = rect.get_height()
        condition = condition_order[idx]
        mean = results[condition]['mean']
        ci_lower = results[condition]['ci_lower']
        ci_upper = results[condition]['ci_upper']
        ax.text(rect.get_x() + rect.get_width()/2., height + 1.,
               f'{mean:.1f}%\n({ci_lower:.1f}%, {ci_upper:.1f}%)',
               ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'{model_name} Prompt Sensitivity')
    ax.set_xticks(x)
    ax.set_ylim(40, 90)
    ax.set_xticklabels(condition_order)  # No rotation needed
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return fig, ax

# %%
fig, ax = plot_experiment_performance(claude_results, "Claude 3.5 Sonnet")
fig.savefig('figures/claude_prompt_sensitivity.png', dpi=300, bbox_inches='tight')
fig.show()

# %%
fig, ax = plot_experiment_performance(deepseek_results, "DeepSeek-v3")
fig.savefig('figures/deepseek_prompt_sensitivity.png', dpi=300, bbox_inches='tight')
fig.show()
# %%

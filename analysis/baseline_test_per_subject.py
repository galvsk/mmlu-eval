# %%
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from mmlu_eval.analysis import MMLU_CATEGORY_MAP, bootstrap_by_subject
from mmlu_eval.paths import CLAUDE_LOGS_DIR, DEEPSEEK_LOGS_DIR, FIGURES_DIR

# %% Load baseline test evals for each model
claude = pd.read_parquet(f'{CLAUDE_LOGS_DIR}/baseline_test/results.parquet')
claude = claude[['answer', 'predicted', 'subject']]
deepseek = pd.read_parquet(f'{DEEPSEEK_LOGS_DIR}/baseline_test/results.parquet')
deepseek = deepseek[['answer', 'predicted', 'subject']]

# %% Ensure our coarser mapping is not excluding any original subjects
assert len(np.unique(claude.subject)) == len(MMLU_CATEGORY_MAP)
assert len(np.unique(deepseek.subject)) == len(MMLU_CATEGORY_MAP)

# Remap to coarser grained subjects
claude['subject'] = claude['subject'].map(MMLU_CATEGORY_MAP)
deepseek['subject'] = deepseek['subject'].map(MMLU_CATEGORY_MAP)
print(claude.subject.value_counts(dropna=False))
print(deepseek.subject.value_counts(dropna=False))


# %% Generate bootstrap performance per subject
claude_results = bootstrap_by_subject(claude)
deepseek_results = bootstrap_by_subject(deepseek)

# %%
def plot_subject_performance(results, model_name, figsize=(12, 8)):
    """
    Plot performance by subject for a single model.
    """
    plt.style.use('seaborn-v0_8-colorblind')
    
    # Get unique subjects and sort by sample size
    subjects = sorted(results.keys(), key=lambda x: results[x]['n_samples'], reverse=True)

       
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up positions for bars
    x = np.arange(len(subjects))
    width = 0.6
    
    # Plot bars and error bars (removed color parameter)
    bars = ax.bar(x, 
                 [results[subject]['mean'] for subject in subjects],
                 width,
                 yerr=[[results[subject]['mean'] - results[subject]['ci_lower'] for subject in subjects],
                       [results[subject]['ci_upper'] - results[subject]['mean'] for subject in subjects]],
                 capsize=5)
    
    # Add text labels with mean and CIs
    for idx, rect in enumerate(bars):
        height = rect.get_height()
        subject = subjects[idx]
        mean = results[subject]['mean']
        ci_lower = results[subject]['ci_lower']
        ci_upper = results[subject]['ci_upper']
        n = results[subject]['n_samples']
        ax.text(rect.get_x() + rect.get_width()/2., height + 3.,
               f'{mean:.1f}%\n({ci_lower:.1f}%, {ci_upper:.1f}%)\nn={n}',
               ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'{model_name} Performance by Subject')
    ax.set_xticks(x)
    ax.set_ylim(40, 105)
    ax.set_xticklabels(subjects, rotation=45, ha='right')
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return fig, ax

# %% Save plots for each model
fig, ax = plot_subject_performance(claude_results, 'Claude 3.5 Sonnet')
fig.savefig(f'{FIGURES_DIR}/claude_by_subject.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
fig, ax = plot_subject_performance(deepseek_results, 'DeepSeek-v3')
fig.savefig(f'{FIGURES_DIR}/deepseek_by_subject.png', dpi=300, bbox_inches='tight')
plt.show()

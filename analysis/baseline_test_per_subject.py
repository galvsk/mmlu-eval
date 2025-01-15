# %%
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from utils import MMLU_CATEGORY_MAP

# %%
claude = pd.read_parquet('../claude_logs/baseline_test/results.parquet')
claude = claude[['answer', 'predicted', 'subject']]
deepseek = pd.read_parquet('../deepseek_logs/baseline_test/results.parquet')
deepseek = deepseek[['answer', 'predicted', 'subject']]

# %%
assert len(np.unique(claude.subject)) == len(MMLU_CATEGORY_MAP)
assert len(np.unique(deepseek.subject)) == len(MMLU_CATEGORY_MAP)

# Remap to coarser grained subjects
claude['subject'] = claude['subject'].map(MMLU_CATEGORY_MAP)
deepseek['subject'] = deepseek['subject'].map(MMLU_CATEGORY_MAP)
display(claude.subject.value_counts(dropna=False))
display(deepseek.subject.value_counts(dropna=False))

# %%
def bootstrap_by_subject(df, n_bootstraps=10_000, random_state=666):
    """
    Bootstrap accuracy with 95% CIs for each subject in the test set.
    """
    np.random.seed(random_state)
    subjects = df['subject'].unique()
    results = {subject: [] for subject in subjects}
    
    for _ in range(n_bootstraps):
        for subject in subjects:
            subject_df = df[df['subject'] == subject]
            sample = subject_df.sample(n=len(subject_df), replace=True)
            acc = (sample['predicted'] == sample['answer']).mean() * 100.
            results[subject].append(acc)
    
    final_results = {}
    for subject in subjects:
        scores = np.array(results[subject])
        final_results[subject] = {
            'mean': np.mean(scores),
            'ci_lower': np.percentile(scores, 2.5),
            'ci_upper': np.percentile(scores, 97.5),
            'n_samples': len(df[df['subject'] == subject])
        }
    
    return final_results

# %%
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

# %%
fig, ax = plot_subject_performance(claude_results, 'Claude 3.5 Sonnet')
fig.savefig('claude_by_subject.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
fig, ax = plot_subject_performance(deepseek_results, 'DeepSeek-v3')
fig.savefig('deepseek_by_subject.png', dpi=300, bbox_inches='tight')
plt.show()

# %%

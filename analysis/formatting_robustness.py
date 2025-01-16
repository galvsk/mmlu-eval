# %%
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from mmlu_eval.analysis import MMLU_CATEGORY_MAP, bootstrap_test, analyse_prompt_sensitivity_by_subject


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
    df = df[['answer', 'predicted', 'subject']]
    assert len(np.unique(df.subject)) == len(MMLU_CATEGORY_MAP)
    df['subject'] = df['subject'].map(MMLU_CATEGORY_MAP) 
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
    df = df[['answer', 'predicted', 'subject']]
    assert len(np.unique(df.subject)) == len(MMLU_CATEGORY_MAP)
    df['subject'] = df['subject'].map(MMLU_CATEGORY_MAP) 
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
def plot_subject_sensitivity_dots(sensitivity_results: dict,
                                prompt_variation: str,
                                figsize=(12, 8)):
    """
    Create a dot plot comparing baseline and modified prompt performance.
    Subjects are shown horizontally, with dots connected by lines showing changes.
    """
    plt.style.use('seaborn-v0_8-colorblind')
    
    # Get subjects sorted by sample size
    subjects = sorted(sensitivity_results.keys(),
                     key=lambda x: sensitivity_results[x]['n_samples'],
                     reverse=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up positions for dots
    x = np.arange(len(subjects))
    
    # Plot connecting lines first (behind dots)
    for idx, subject in enumerate(subjects):
        baseline = sensitivity_results[subject]['means'][0]
        modified = sensitivity_results[subject]['means'][1]
        is_sig = sensitivity_results[subject]['is_significant']
        
    # Plot dots
    ax.scatter(x, [sensitivity_results[s]['means'][0] for s in subjects],
               s=80, color='#3182bd', label='Baseline', zorder=2)
    ax.scatter(x, [sensitivity_results[s]['means'][1] for s in subjects],
               s=80, color='#31a354', label=prompt_variation, zorder=2)
    
    # Add error bars
    for idx, subject in enumerate(subjects):
        # Baseline error bars
        ax.errorbar(idx, sensitivity_results[subject]['means'][0],
                   yerr=[[sensitivity_results[subject]['means'][0] - sensitivity_results[subject]['lower_cis'][0]],
                         [sensitivity_results[subject]['upper_cis'][0] - sensitivity_results[subject]['means'][0]]],
                   color='#3182bd', capsize=3, alpha=0.5, zorder=1)
        
        # Modified error bars
        ax.errorbar(idx, sensitivity_results[subject]['means'][1],
                   yerr=[[sensitivity_results[subject]['means'][1] - sensitivity_results[subject]['lower_cis'][1]],
                         [sensitivity_results[subject]['upper_cis'][1] - sensitivity_results[subject]['means'][1]]],
                   color='#31a354', capsize=3, alpha=0.5, zorder=1)
        
        # Add delta labels for significant changes only
        delta = sensitivity_results[subject]['delta']
        if sensitivity_results[subject]['is_significant']:
            ax.text(idx, max(sensitivity_results[subject]['means']) + 4,
                   f'{delta:+.1f}%*',
                   ha='center', va='bottom', fontsize=12)
        else:
             ax.text(idx, max(sensitivity_results[subject]['means']) + 3,
                   f'{delta:+.1f}%',
                   ha='center', va='bottom', fontsize=12)
   
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'{prompt_variation} Impact on Performance by Subject')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha='right')
    
    # Set y-axis limits with padding for labels
    ymin = min([min(r['lower_cis']) for r in sensitivity_results.values()]) - 5
    ymax = max([max(r['upper_cis']) for r in sensitivity_results.values()]) + 10
    ax.set_ylim(ymin, ymax)
    
    ax.legend()
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return fig, ax

# %% Compare baseline to uppercase for Claude
upper_claude = analyse_prompt_sensitivity_by_subject(claude_dfs['Baseline'], claude_dfs['Uppercase'])

# %%
fig, ax = plot_subject_sensitivity_dots(upper_claude, prompt_variation='Uppercase')
fig.savefig('figures/claude_uppercase.png', dpi=300, bbox_inches='tight')
fig.show()

# %% Compare baseline to duplicate for Claude
dup_claude = analyse_prompt_sensitivity_by_subject(claude_dfs['Baseline'], claude_dfs['Duplicate wrongs'])

# %%
fig, ax = plot_subject_sensitivity_dots(dup_claude, prompt_variation='Duplicate Wrongs')
fig.savefig('figures/claude_duplicate.png', dpi=300, bbox_inches='tight')
fig.show()

# %% Compare baseline to randomcase for Deepseek
random_deepseek = analyse_prompt_sensitivity_by_subject(deepseek_dfs['Baseline'], deepseek_dfs['Random case'])

# %%
fig, ax = plot_subject_sensitivity_dots(random_deepseek, prompt_variation='Random case')
fig.savefig('figures/deepseek_random.png', dpi=300, bbox_inches='tight')
fig.show()

# %%

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
import seaborn as sns
from typing import Dict, List, Optional
from mmlu_eval.analysis import bootstrap_test, bootstrap_by_subject, MMLU_CATEGORY_MAP
from mmlu_eval.paths import CLAUDE_LOGS_DIR, DEEPSEEK_LOGS_DIR, FIGURES_DIR


# %% Original eval on entire MMLU, so ensure all experiments are on identical subsets of data
def harmonise_datasets(dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Ensure all dataframes in the dictionary contain the same subset of questions.
    """
    # Get common questions across all datasets
    common_questions = set.intersection(*[set(df['question']) for df in dfs.values()])
    print(f"Number of common questions across datasets: {len(common_questions)}")
    
    # Filter each dataset to only include common questions
    harmonised_dfs = {
        condition: df[df['question'].isin(common_questions)].copy()
        for condition, df in dfs.items()
    }
    
    return harmonised_dfs

# %% Load in relevant experiment results dataframes
paths = {'Reference' : 'permuted_test',
         'Claude Generated' : 'claude_generated_test',
         'Deepseek Generated': 'deepseek_generated_test'}

claude_dfs = {}
for exp, path in paths.items():
    df = pd.read_parquet(os.path.join(CLAUDE_LOGS_DIR, path, 'results.parquet'))
    if exp == 'Reference':
        df['subject'] = df['subject'].map(MMLU_CATEGORY_MAP) 
    df = df[df.subject.isin(['STEM', 'Social Sciences', 'Law & Ethics'])]
    claude_dfs[exp] = df


deepseek_dfs = {}
for exp, path in paths.items():
    df = pd.read_parquet(os.path.join(DEEPSEEK_LOGS_DIR, path, 'results.parquet'))
    if exp == 'Reference':
       df['subject'] = df['subject'].map(MMLU_CATEGORY_MAP) 
    df = df[df.subject.isin(['STEM', 'Social Sciences', 'Law & Ethics'])]
    deepseek_dfs[exp] = df


# %% Harmonise
claude_dfs = harmonise_datasets(claude_dfs)
deepseek_dfs = harmonise_datasets(deepseek_dfs)

# %% Get bootstrap results
claude_results = {}
for exp, df in claude_dfs.items():
    claude_results[exp] = bootstrap_test(df)

deepseek_results = {}
for exp, df in deepseek_dfs.items():
    deepseek_results[exp] = bootstrap_test(df)

# %%
def plot_experiment_performance(results: Dict[str, Dict], model_name: str, figsize: tuple = (10, 6)) -> tuple:
    """
    Plot performance across different experimental conditions for a single model.
    """
    plt.style.use('seaborn-v0_8-colorblind')
    
    # Define the order of conditions
    condition_order = ['Reference', 'Claude Generated', 'Deepseek Generated']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up positions for bars
    x = np.arange(len(condition_order))
    width = 0.5
    
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
        ax.text(rect.get_x() + rect.get_width()/2., height + 2.,
               f'{mean:.1f}%\n({ci_lower:.1f}%, {ci_upper:.1f}%)',
               ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'{model_name} Performance on MMLU Variants')
    ax.set_xticks(x)
    ax.set_ylim(40, 90)
    ax.set_xticklabels(condition_order, rotation=15)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return fig, ax

# %%
fig, ax = plot_experiment_performance(claude_results, model_name='Claude 3.5 Sonnet')
fig.savefig(f"{FIGURES_DIR}/claude_on_generateds.png", dpi=300, bbox_inches='tight')
fig.show()


# %%
fig, ax = plot_experiment_performance(deepseek_results, model_name='DeepSeek-v3')
fig.savefig(f"{FIGURES_DIR}/deepseek_on_generateds.png", dpi=300, bbox_inches='tight')
fig.show()

# %% Get results per subject
claude_per_subject = {}
for exp, df in claude_dfs.items():
   claude_per_subject[exp] = bootstrap_by_subject(df)

deepseek_per_subject = {}
for exp, df in deepseek_dfs.items():
   deepseek_per_subject[exp] = bootstrap_by_subject(df)

# %%
def plot_subject_performance(subject_results: Dict[str, Dict[str, Dict]], 
                           model_name: str,
                           figsize: tuple = (12, 6)) -> tuple:
    """
    Create a dot plot comparing performance across conditions by subject.
    
    Parameters:
    -----------
    subject_results : Dict[str, Dict[str, Dict]]
        Dictionary with structure:
        {condition: {subject: {mean, ci_lower, ci_upper, n_samples}}}
    model_name : str
        Name of the model for the plot title
    """
    plt.style.use('seaborn-v0_8-colorblind')
    
    # Define the order of conditions and their colors
    condition_order = ['Reference', 'Claude Generated', 'Deepseek Generated']
    condition_colors = {
        'Reference': '#3182bd',
        'Claude Generated': '#31a354',
        'Deepseek Generated': '#e6550d'
    }
    
    # Get subjects sorted by sample size (using Reference condition)
    subjects = sorted(subject_results['Reference'].keys(),
                     key=lambda x: subject_results['Reference'][x]['n_samples'],
                     reverse=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up positions for dots
    x = np.arange(len(subjects))
    width = 0.2  # Spacing between dots for same subject
    
    # Plot dots and error bars for each condition
    for i, (condition, color) in enumerate(zip(condition_order, condition_colors.values())):
        offset = (i - 1) * width  # Center the dots around tick marks
        
        # Plot dots
        ax.scatter(x + offset, 
                  [subject_results[condition][s]['mean'] for s in subjects],
                  s=80, color=color, label=condition, zorder=2)
        
        # Add error bars
        for idx, subject in enumerate(subjects):
            result = subject_results[condition][subject]
            ax.errorbar(idx + offset, result['mean'],
                       yerr=[[result['mean'] - result['ci_lower']],
                             [result['ci_upper'] - result['mean']]],
                       color=color, capsize=3, alpha=0.5, zorder=1)
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'{model_name} Performance by Subject')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha='right')
    
    # Add sample size information below subject names with more space
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for idx, subject in enumerate(subjects):
        ax.text(idx, -0.15, 
                f'n={subject_results["Reference"][subject]["n_samples"]}',
                ha='center', va='top', fontsize=8,
                transform=trans) 
    
    ax.set_ylim(40, 95)
    ax.legend(loc='lower right')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return fig, ax

#%%
fig, ax = plot_subject_performance(claude_per_subject, model_name='Claude 3.5 Sonnet')
fig.savefig(f"{FIGURES_DIR}/claude_on_generated_per_subject.png", dpi=300, bbox_inches='tight')
fig.show()

# %%
fig, ax = plot_subject_performance(deepseek_per_subject, model_name='DeepSeek-v3')
fig.savefig(f"{FIGURES_DIR}/deepseek_on_generated_per_subject.png", dpi=300, bbox_inches='tight')
fig.show()

# %%

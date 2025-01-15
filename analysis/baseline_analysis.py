# %%
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

claude_paths = ['../claude_logs/baseline_train', '../claude_logs/baseline_test']
deepseek_paths = ['../deepseek_logs/baseline_train', '../deepseek_logs/baseline_test']

# %%
claude_dfs = []
for path in claude_paths:
    df = pd.read_parquet(os.path.join(path, 'results.parquet'))
    df = df[['predicted', 'answer', 'fold']]
    claude_dfs.append(df)

deepseek_dfs = []
for path in deepseek_paths:
    df = pd.read_parquet(os.path.join(path, 'results.parquet'))
    df = df[['predicted', 'answer', 'fold']]
    deepseek_dfs.append(df)
    
deepseek = pd.concat([*deepseek_dfs])
claude = pd.concat([*claude_dfs])

# %%
def plot_answer_distributions(claude_df, deepseek_df, fold='test', figsize=(12, 6)):
    """
    Plot the distribution of predicted answers for Claude and Deepseek, compared to ground truth.
    
    Parameters:
    -----------
    claude_df : pandas.DataFrame
        DataFrame containing Claude's predictions with columns ['predicted', 'answer', 'fold']
    deepseek_df : pandas.DataFrame
        DataFrame containing Deepseek's predictions with columns ['predicted', 'answer', 'fold']
    fold : str, optional (default='test')
        Which fold to plot ('train' or 'test')
    figsize : tuple, optional (default=(12, 6))
        Size of the figure
    """
    # Create a mapping dictionary
    num_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    
    # Filter out -1 and convert to new DataFrames to avoid modifying originals
    claude_subset = claude_df[
        (claude_df['fold'] == fold) & 
        (claude_df['predicted'] != -1)
    ].copy()
    
    deepseek_subset = deepseek_df[
        (deepseek_df['fold'] == fold) & 
        (deepseek_df['predicted'] != -1)
    ].copy()
    
    # Convert numeric answers to letters
    claude_subset['predicted'] = claude_subset['predicted'].map(num_to_letter)
    claude_subset['answer'] = claude_subset['answer'].map(num_to_letter)
    deepseek_subset['predicted'] = deepseek_subset['predicted'].map(num_to_letter)
    
    # Create separate counts for each distribution
    claude_counts = pd.Series(claude_subset['predicted'].value_counts()).reindex(['A', 'B', 'C', 'D']).fillna(0)
    deepseek_counts = pd.Series(deepseek_subset['predicted'].value_counts()).reindex(['A', 'B', 'C', 'D']).fillna(0)
    ground_truth_counts = pd.Series(claude_subset['answer'].value_counts()).reindex(['A', 'B', 'C', 'D']).fillna(0)
    
    # Convert to percentages
    total_samples = len(claude_subset)
    claude_pct = (claude_counts / total_samples) * 100
    deepseek_pct = (deepseek_counts / total_samples) * 100
    ground_truth_pct = (ground_truth_counts / total_samples) * 100
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8-colorblind')
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set the width of each bar and positions of the bars
    width = 0.25
    x = np.arange(len(['A', 'B', 'C', 'D']))
    
    # Create the bars
    bars1 = ax.bar(x - width, claude_pct, width, label='Claude 3.5 Sonnet', color='#8c96c6', alpha=0.8)
    bars2 = ax.bar(x, deepseek_pct, width, label='DeepSeek-v3', color='#88419d', alpha=0.8)
    bars3 = ax.bar(x + width, ground_truth_pct, width, label='Ground Truth', color='#4daf4a', alpha=0.8)
    
    # Customize the plot
    ax.set_ylabel('Percentage (%)')
    ax.set_title(f'Distribution of MMLU Answers ({fold.capitalize()} Set)')
    ax.set_xticks(x)
    ax.set_xticklabels(['A', 'B', 'C', 'D'])
    ax.legend()
    
    # Add value labels on top of each bar
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', rotation=0)
    
    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)
    
    # Add grid for better readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return fig, ax

# %%
# Example usage:
fig, ax = plot_answer_distributions(claude, deepseek, fold='test')
fig.savefig('baseline_test_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
fig, ax = plot_answer_distributions(claude, deepseek, fold='train')
fig.savefig('baseline_train_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
def bootstrap_performance(df, n_bootstraps=10_000, random_state=42):
    """
    Bootstrap accuracy with 95% CIs for train and test sets.
    """
    np.random.seed(random_state)
    train_results, test_results = [], []
    
    train_df = df[df['fold'] == 'train']
    test_df = df[df['fold'] == 'test']
    
    for _ in range(n_bootstraps):
        # Sample both train and test in same loop iteration
        train_sample = train_df.sample(n=len(train_df), replace=True)
        test_sample = test_df.sample(n=len(test_df), replace=True)
        
        train_acc = (train_sample['predicted'] == train_sample['answer']).mean() * 100.
        test_acc = (test_sample['predicted'] == test_sample['answer']).mean() * 100.
        
        train_results.append(train_acc)
        test_results.append(test_acc)
    
    return {
        'train': {
            'mean': np.mean(train_results),
            'ci_lower': np.percentile(train_results, 2.5),
            'ci_upper': np.percentile(train_results, 97.5)
        },
        'test': {
            'mean': np.mean(test_results),
            'ci_lower': np.percentile(test_results, 2.5),
            'ci_upper': np.percentile(test_results, 97.5)
        }
    }

# %%
claude_acc = bootstrap_performance(claude)
deepseek_acc = bootstrap_performance(deepseek)

# %%
def plot_bootstrap_results(claude_results, deepseek_results, figsize=(10, 6)):
    """
    Plot bootstrap results for both models showing train/test accuracy with CIs.
    """
    plt.style.use('seaborn-v0_8-colorblind')
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up positions for bars
    x = np.arange(2)
    width = 0.35
    
    # Plot bars and error bars for both models
    claude_bars = ax.bar(x - width/2, 
          [claude_results['train']['mean'], claude_results['test']['mean']], 
          width,
          label='Claude 3.5 Sonnet',
          yerr=[[claude_results['train']['mean'] - claude_results['train']['ci_lower'],
                claude_results['test']['mean'] - claude_results['test']['ci_lower']],
               [claude_results['train']['ci_upper'] - claude_results['train']['mean'],
                claude_results['test']['ci_upper'] - claude_results['test']['mean']]],
          capsize=5)
    
    deepseek_bars = ax.bar(x + width/2, 
          [deepseek_results['train']['mean'], deepseek_results['test']['mean']], 
          width,
          label='DeepSeek-v3',
          yerr=[[deepseek_results['train']['mean'] - deepseek_results['train']['ci_lower'],
                deepseek_results['test']['mean'] - deepseek_results['test']['ci_lower']],
               [deepseek_results['train']['ci_upper'] - deepseek_results['train']['mean'],
                deepseek_results['test']['ci_upper'] - deepseek_results['test']['mean']]],
          capsize=5)
    
    # Add text labels with mean and CIs
    def add_labels(bars, results):
        for idx, rect in enumerate(bars):
            height = rect.get_height()
            fold = 'train' if idx == 0 else 'test'
            mean = results[fold]['mean']
            ci_lower = results[fold]['ci_lower']
            ci_upper = results[fold]['ci_upper']
            ax.text(rect.get_x() + rect.get_width()/2., height + 1.,
                   f'{mean:.1f}%\n({ci_lower:.1f}%, {ci_upper:.1f}%)',
                   ha='center', va='bottom')
    
    add_labels(claude_bars, claude_results)
    add_labels(deepseek_bars, deepseek_results)
    
    # Customize plot
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Performance with 95% CIs')
    ax.set_xticks(x)
    ax.set_xticklabels(['Train', 'Test'])
    ax.set_ylim(50, 97)
    ax.legend()
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax

# %%
# Example usage:
fig, ax = plot_bootstrap_results(claude_acc, deepseek_acc)
fig.savefig('baseline_bootstrap_results.png', dpi=300, bbox_inches='tight')
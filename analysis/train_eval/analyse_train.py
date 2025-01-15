# %%
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt



# %%
claude_trn = pd.read_parquet('../claude_logs/baseline_train/results.parquet')
deepseek_trn = pd.read_parquet('../deepseek_logs/baseline_train/results.parquet')


# %%
# Deepseek was significantly worse at following the instruction to only output A-D
# Thus we saved responses, and can do some straightforward formatting

def format_deepseek_responses(response: str) -> str:
    """
    Extracts valid MMLU response (A-D) if present at start of string, mapped a 0-4 indices.
    Returns 'INVALID' if no valid response found.
    """
    dict_map = {'A' : 0, 'B': 1, 'C': 2, 'D': 3, 'refusal': -1}
    if pd.isna(response):
        return 'INVALID'
    if response in ['A', 'B', 'C', 'D', 'refusal']:
        return dict_map[response]
       
    match = re.match(r'^[^A-D]*([A-D])', response, re.IGNORECASE)
    return dict_map[match.group(1).upper()] if match else 'INVALID'

deepseek_trn['classified'] = deepseek_trn['response'].apply(format_deepseek_responses)


# %%
pd.crosstab(deepseek_trn['predicted'], deepseek_trn['classified'])

# %%
# Overwrite the predicted responses if they were successfully parsed as 0-3
valid_values = [0, 1, 2, 3]
mask = deepseek_trn['classified'].isin(valid_values)
deepseek_trn.loc[mask, 'predicted'] = deepseek_trn.loc[mask, 'classified']

# %%
pd.crosstab(deepseek_trn['predicted'], deepseek_trn['classified'])


# %%
display(deepseek_trn.predicted.value_counts(dropna=False)[[0, 1, 2, 3, -1]])
display(claude_trn.predicted.value_counts(dropna=False)[[0, 1, 2, 3, -1]])

# %%
def plot_mmlu_distributions(pred1, pred2, figsize=(10, 6)):
    """
    Plot MMLU answer distributions for Claude 3.5 Sonnet and DeepSeek v3 in a single plot.
    """
    # Filter and map predictions
    category_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    pred1_mapped = pred1[pred1 != -1].map(category_map)
    pred2_mapped = pred2[pred2 != -1].map(category_map)
    
    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        'Answer': pd.concat([pred1_mapped, pred2_mapped]),
        'Model': ['Claude 3.5 Sonnet'] * len(pred1_mapped) + ['DeepSeek v3'] * len(pred2_mapped)
    })
    
    # Setup plot
    plt.figure(figsize=figsize)
    
    # Create grouped bar plot
    ax = sns.countplot(data=plot_data, x='Answer', hue='Model', 
                      order=['A', 'B', 'C', 'D'],
                      palette=['#4878CF', '#6ACC65'])
    
    # Explicitly set x-axis ticks and labels
    ax.set_xticks(range(4))
    ax.set_xticklabels(['A', 'B', 'C', 'D'])
    
    plt.title('Answer distributions on MMLU Eval', fontsize=14, pad=10)
    plt.xlabel('Answer Categories', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add value labels
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}',
                   xy=(p.get_x() + p.get_width()/2, height),
                   xytext=(0, 3),
                   textcoords='offset points',
                   ha='center')
    
    plt.tight_layout()
    return plt.gcf()

# Usage:
fig = plot_mmlu_distributions(claude_trn['predicted'], deepseek_trn['predicted'])
fig.savefig('trn_answer_distributions.png')
plt.show()


# %%
def sample_accuracy_with_ci(df, num_samples=1000, sample_size=None, confidence_level=0.95, random_state=666):
    """
    Calculate accuracy statistics using sampling with replacement.
    
    Parameters:
    df: DataFrame with 'predicted' and 'answer' columns
    num_samples: Number of sampling iterations
    sample_size: Size of each sample (default: same as original data)
    confidence_level: Confidence level for intervals
    random_state: Random seed for reproducibility
    
    Returns:
    dict with mean accuracy, CI bounds, and all samples
    """
    np.random.seed(random_state)
    
    if sample_size is None:
        sample_size = len(df)
    
    # Get binary outcomes (correct/incorrect)
    outcomes = (df['predicted'] == df['answer']).astype(int)
    
    # Perform sampling with replacement
    accuracies = []
    for _ in range(num_samples):
        sample = np.random.choice(outcomes, size=sample_size, replace=True)
        accuracy = np.mean(sample) * 100.
        accuracies.append(accuracy)
    
    accuracies = np.array(accuracies)
    
    return {
        'mean': np.mean(accuracies),
        'ci_lower': np.percentile(accuracies, (1 - confidence_level) * 100 / 2),
        'ci_upper': np.percentile(accuracies, (1 + confidence_level) * 100 / 2),
        'samples': accuracies
    }


def plot_model_comparison(df1, df2, num_samples=1000, sample_size=None, confidence_level=0.95):
    """
    Create a vertical comparison plot with means and confidence intervals.
    """
    sonnet_stats = sample_accuracy_with_ci(df1, num_samples, sample_size, confidence_level)
    deepseek_stats = sample_accuracy_with_ci(df2, num_samples, sample_size, confidence_level)
    
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Set background color
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Plot settings
    models = ['Claude 3.5 Sonnet', 'Deepseek v3']
    colors = ['#4878CF', '#6ACC65']
    x_positions = [0, 1]
    
    # Plot data
    for i, (model, stats, color) in enumerate(zip(models, 
                                                [sonnet_stats, deepseek_stats],
                                                colors)):
        # Plot confidence interval
        ax.vlines(x=x_positions[i], ymin=stats['ci_lower'], ymax=stats['ci_upper'],
                 color=color, linewidth=2)
        
        # Add caps to the CI lines
        ax.hlines(y=[stats['ci_lower'], stats['ci_upper']], 
                 xmin=x_positions[i]-0.1, xmax=x_positions[i]+0.1,
                 color=color, linewidth=2)
        
        # Plot mean point
        ax.plot(x_positions[i], stats['mean'], 'o', color=color, 
                markersize=10, markeredgecolor='black', markeredgewidth=1.5,
                label=f'{model}: {stats["mean"]:.1f}% [{stats["ci_lower"]:.1f}%, {stats["ci_upper"]:.1f}%]')
        
    
    ax.set_ylabel('Accuracy (%)', labelpad=10, fontsize=12)
    
    # Set x-axis ticks and labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models, fontsize=10)
    
    # Add grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Set axis limits
    ax.set_ylim(min(sonnet_stats['ci_lower'], deepseek_stats['ci_lower']) - 0.5,
                max(sonnet_stats['ci_upper'], deepseek_stats['ci_upper']) + 0.5)
    ax.set_xlim(-0.5, 1.5)
    
    # Add legend in lower right
    ax.legend(loc='lower right', fontsize=10, frameon=False)
    
    # Adjust layout
    fig.tight_layout()
    
    stats_dict = {
        'Claude 3.5': sonnet_stats,
        'Deepseek v3': deepseek_stats
    }
    
    return fig, stats_dict

fig, stats = plot_model_comparison(claude_trn, deepseek_trn, num_samples=10000)
fig.savefig('trn_accuracies.png', dpi=300, bbox_inches='tight')
fig.show()
# %%

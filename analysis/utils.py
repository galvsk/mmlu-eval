import numpy as np
import pandas as pd


def bootstrap_train_vs_test_performance(df, n_bootstraps=10_000, random_state=666):
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

def bootstrap_test(df, n_bootstraps=10_000, random_state=666):
    """
    Bootstrap accuracy with 95% CIs for test set.
    """
    np.random.seed(random_state)
    results = [] 
    for _ in range(n_bootstraps):
        # Sample both train and test in same loop iteration
        sample = df.sample(n=len(df), replace=True)
        acc = (sample['predicted'] == sample['answer']).mean() * 100.
        results.append(acc)
    
    return {
        'mean': np.mean(results),
        'ci_lower': np.percentile(results, 2.5),
        'ci_upper': np.percentile(results, 97.5)
    }


# Define coarser grained subject mapping to more easily sub-stratify model performance
MMLU_CATEGORY_MAP = {
    # STEM
    'elementary_mathematics': 'STEM',
    'high_school_mathematics': 'STEM',
    'college_mathematics': 'STEM',
    'abstract_algebra': 'STEM',
    'high_school_physics': 'STEM',
    'college_physics': 'STEM',
    'conceptual_physics': 'STEM',
    'high_school_chemistry': 'STEM',
    'college_chemistry': 'STEM',
    'high_school_biology': 'STEM',
    'college_biology': 'STEM',
    'astronomy': 'STEM',
    'electrical_engineering': 'STEM',
    'high_school_computer_science': 'STEM',
    'college_computer_science': 'STEM',
    'computer_security': 'STEM',
    'machine_learning': 'STEM',
    'high_school_statistics': 'STEM',

    # Social Sciences
    'high_school_psychology': 'Social Sciences',
    'professional_psychology': 'Social Sciences',
    'sociology': 'Social Sciences',
    'high_school_geography': 'Social Sciences',
    'high_school_government_and_politics': 'Social Sciences',
    'high_school_macroeconomics': 'Social Sciences',
    'high_school_microeconomics': 'Social Sciences',
    'econometrics': 'Social Sciences',

    # Business & Professional
    'professional_accounting': 'Business & Professional',
    'marketing': 'Business & Professional',
    'management': 'Business & Professional',
    'public_relations': 'Business & Professional',
    'business_ethics': 'Business & Professional',

    # Medicine & Health
    'professional_medicine': 'Medicine & Health',
    'college_medicine': 'Medicine & Health',
    'clinical_knowledge': 'Medicine & Health',
    'anatomy': 'Medicine & Health',
    'medical_genetics': 'Medicine & Health',
    'human_aging': 'Medicine & Health',
    'human_sexuality': 'Medicine & Health',
    'nutrition': 'Medicine & Health',
    'virology': 'Medicine & Health',

    # Law & Ethics
    'professional_law': 'Law & Ethics',
    'international_law': 'Law & Ethics',
    'jurisprudence': 'Law & Ethics',
    'moral_scenarios': 'Law & Ethics',
    'moral_disputes': 'Law & Ethics',

    # Humanities
    'philosophy': 'Humanities',
    'world_religions': 'Humanities',
    'formal_logic': 'Humanities',
    'logical_fallacies': 'Humanities',
    'high_school_world_history': 'Humanities',
    'high_school_us_history': 'Humanities',
    'high_school_european_history': 'Humanities',
    'prehistory': 'Humanities',

    # Global Studies
    'security_studies': 'Global Studies',
    'us_foreign_policy': 'Global Studies',
    'global_facts': 'Global Studies',

    # Other
    'miscellaneous': 'Other'
}
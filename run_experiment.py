#!/usr/bin/env python3
import argparse
from textwrap import dedent
from mmlu_experimenter import MMLUExperimenter
from mmlu_formatter import MMLUPromptDefault, MMLUPromptPermuted, MMLUPromptUpperCase, MMLUPromptRandomCase


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run MMLU experiments with Claude',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent('''
            Example usage:
            python run_experiment.py \\
                --exp-path experiments/baseline \\
                --df-path ref_dataframes/mmlu_test.parquet \\
                --desc "Baseline test run" \\
                --max-questions 100 \\
                --prompt-style permuted
        ''')
    )
    
    parser.add_argument(
        '--exp-path',
        type=str,
        required=True,
        help='Path to store experiment results'
    )
    
    parser.add_argument(
        '--df-path',
        type=str,
        required=True,
        help='Path to MMLU dataset parquet file'
    )
    
    parser.add_argument(
        '--desc',
        type=str,
        default=None,
        help='Description of the experiment (required for new experiments)'
    )
    
    parser.add_argument(
        '--max-questions',
        type=int,
        default=None,
        help='Maximum number of questions to process'
    )
    
    parser.add_argument(
        '--save-frequency',
        type=int,
        default=10,
        help='How often to save results (in number of questions)'
    )

    parser.add_argument(
        '--prompt-style',
        type=str,
        choices=['default', 'permuted', 'uppercase', 'randomcase'],
        default='default',
        help='Style of prompt formatting to use'
    )    
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Select prompt style
    prompt_style = {
        'default': MMLUPromptDefault,
        'permuted': MMLUPromptPermuted,
        'uppercase': MMLUPromptUpperCase,
        'randomcase': MMLUPromptRandomCase
    }[args.prompt_style] 
    
    # Initialize experimenter
    experimenter = MMLUExperimenter(
        experiment_path=args.exp_path,
        df_path=args.df_path,
        description=args.desc,
        save_frequency=args.save_frequency,
        prompt_style=prompt_style
    )
    
    # Run experiment
    experimenter.run_experiment(
        max_questions=args.max_questions,
    )
    
    # Print results
    results = experimenter.get_results_summary()
    print("\nExperiment Results:")
    print(f"Description: {results['description']}")
    print(f"Model: {results['model']}")
    print(f"Prompt Style: {args.prompt_style}")
    print(f"Questions Completed: {results['completed_questions']}/{results['total_questions']}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Last Updated: {results['last_updated']}")

if __name__ == "__main__":
    main()

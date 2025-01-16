import argparse
from textwrap import dedent
from mmlu_eval.experimenter import MMLUExperimenter
from mmlu_eval.formatter import (
    MMLUPromptDefault, 
    MMLUPromptPermuted, 
    MMLUPromptUpperCase,
    MMLUPromptRandomCase, 
    MMLUPromptDuplicateWrong
)
from mmlu_eval.paths import (
    get_ref_data_path,
    get_experiment_path,
    MMLU_TEST_FILE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run MMLU experiments with Claude',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent('''
            Example usage:
            python run_mmlu_eval.py \\
                --experiment baseline \\
                --desc "Baseline test run" \\
                --max-questions 100 \\
                --api claude \\
                --prompt-style permuted
        ''')
    )

    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        help='Name of experiment directory to create in logs'
    )

    parser.add_argument(
        '--test-data',
        type=str,
        default=MMLU_TEST_FILE,
        help='Name of test data file in ref_dataframes'
    )

    parser.add_argument(
        '--desc',
        type=str,
        required=True,
        help='Description of the experiment'
    )

    parser.add_argument(
        '--max-questions',
        type=int,
        default=None,
        help='Maximum number of questions to process'
    )

    parser.add_argument(
        '--api',
        type=str,
        choices=['claude', 'deepseek'],
        default='claude',
        help='Which model API to use'
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
        choices=['default', 'permuted', 'uppercase', 'randomcase', 'duplicatewrong'],
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
        'randomcase': MMLUPromptRandomCase,
        'duplicatewrong': MMLUPromptDuplicateWrong
    }[args.prompt_style]

    # Initialize experimenter
    experimenter = MMLUExperimenter(
        experiment_path=get_experiment_path(args.experiment, args.api),
        df_path=get_ref_data_path(args.test_data),
        api=args.api,
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
    print(f"Model: {results['model']['type']}")
    print(f"Data: {args.test_data}")
    print(f"Prompt Style: {args.prompt_style}")
    print(f"Questions Completed: {results['completed_questions']}/{results['total_questions']}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Last Updated: {results['last_updated']}")
    print(f"\nResults saved to: {get_experiment_path(args.experiment, args.api)}")

if __name__ == "__main__":
    main()

import argparse
from textwrap import dedent
from mmlu_eval.paths import get_experiment_path, MMLU_TEST_FILE
from mmlu_eval.experimenter import MMLUExperimenter
from mmlu_eval.alternative_experimenter import AlternativeExperimenter
from mmlu_eval.formatter import (
    MMLUPromptDefault, 
    MMLUPromptPermuted, 
    MMLUPromptUpperCase,
    MMLUPromptRandomCase, 
    MMLUPromptDuplicateWrong
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run MMLU evaluation experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent('''
            Example usage:
            # Standard MMLU evaluation
            python run_mmlu_eval.py \\
                --experiment baseline_test \\
                --df-path data/ref_dataframes/mmlu_test.parquet \\
                --desc "Claude baseline test eval for 100 questions" \\
                --max-questions 100 \\
                --api claude \\
                --prompt-style permuted

            # Alternative answers evaluation
            python run_mmlu_eval.py \\
                --experiment alternative_baseline \\
                --df-path data/generated_dataframes/deepseek_generated_dataframe.parquet \\
                --desc "Deepseek evaluation using deepseek generated incorrect answers" \\
                --api deepseek \\
                --alternative-mode generated_only
        ''')
    )

    # Original arguments
    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        help='Name of experiment'
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
        default=100,
        help='How often to save results (in number of questions)'
    )

    # Add alternative evaluation arguments
    parser.add_argument(
        '--alternative-mode',
        type=str,
        choices=['generated_only', 'all_answers', None],
        default=None,
        help='If set, run alternative answers evaluation in specified mode'
    )

    parser.add_argument(
        '--df-path',
        type=str,
        default=MMLU_TEST_FILE,
        help='Path to evaluation dataframe (default: standard MMLU test set)'
    )

    # Prompt style only used for standard MMLU
    parser.add_argument(
        '--prompt-style',
        type=str,
        choices=['default', 'permuted', 'uppercase', 'randomcase', 'duplicatewrong'],
        default='default',
        help='Style of prompt formatting (only used for standard MMLUExperimenter)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Determine which evaluation mode to use
    if args.alternative_mode:
        print("\nNote: --prompt-style argument is ignored when running alternative evaluation")
        
        # Run alternative answers evaluation
        experimenter = AlternativeExperimenter(
            experiment_path=get_experiment_path(args.experiment, args.api),
            df_path=args.df_path,
            api=args.api,
            description=args.desc,
            save_frequency=args.save_frequency,
            eval_mode=args.alternative_mode
        )
    else:
        # Run standard MMLU evaluation
        # Select prompt style
        prompt_style = {
            'default': MMLUPromptDefault,
            'permuted': MMLUPromptPermuted,
            'uppercase': MMLUPromptUpperCase,
            'randomcase': MMLUPromptRandomCase,
            'duplicatewrong': MMLUPromptDuplicateWrong
        }[args.prompt_style]

        experimenter = MMLUExperimenter(
            experiment_path=get_experiment_path(args.experiment, args.api),
            df_path=args.df_path,
            api=args.api,
            description=args.desc,
            save_frequency=args.save_frequency,
            prompt_style=prompt_style
        )

    # Run experiment
    experimenter.run_experiment(max_questions=args.max_questions)

    # Print results
    results = experimenter.get_results_summary()
    print("\nExperiment Results:")
    print(f"Description: {results['description']}")
    print(f"Model: {results['model']['type']}")
    
    if args.alternative_mode:
        print(f"Evaluation Mode: {results['eval_mode']}")
    else:
        print(f"Prompt Style: {args.prompt_style}")
        
    print(f"Questions Completed: {results['completed_questions']}/{results['total_questions']}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"\nLast Updated: {results['last_updated']}")
    print(f"\nResults saved to: {get_experiment_path(args.experiment, args.api)}")


if __name__ == '__main__':
    main()

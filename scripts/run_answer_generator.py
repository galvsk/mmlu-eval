import argparse
from textwrap import dedent
from mmlu_eval.answer_generator import DatasetGenerator, AlternativeAnswerConfig
from mmlu_eval.paths import MMLU_TEST_FILE


def parse_args():
    """Parse command line arguments."""
    example_usage = (
        'Example usage:\n'
        'python generate_model_answers.py \\\n'
        '    --df-path ref_dataframes/mmlu_test.parquet \\\n'
        '    --api claude \\\n'
        '    --seed 123'
    )
    
    parser = argparse.ArgumentParser(
        description='Generate alternative answers for MMLU questions using LLMs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(example_usage)
    )

    parser.add_argument(
        '--df-path',
        type=str,
        default=MMLU_TEST_FILE,
        help='Path to MMLU dataset parquet file (default: MMLU test set)'
    )

    parser.add_argument(
        '--api',
        type=str,
        choices=['claude', 'deepseek'],
        default='claude',
        help='Which model API to use for generating answers'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=666,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--save-frequency',
        type=int,
        default=100,
        help='How often to save results (in number of questions)'
    )

    
    return parser.parse_args()

def print_generation_summary(df):
    """Print summary of the generation process."""
    print(f'\nGeneration complete:')
    print(f'- Generated {len(df)} questions')
    
    if 'subject' in df.columns:
        subjects = df['subject'].value_counts()
        print('\nSubject distribution:')
        for subject, count in subjects.items():
            print(f'- {subject}: {count}')


def main():
    """Main execution function."""
    args = parse_args()

    # Initialize generator
    generator = DatasetGenerator(
        df_path=args.df_path,
        api=args.api,
        seed=args.seed,
        save_frequency=args.save_frequency
    )

    # Print initial setup
    print(f'\nGenerating alternative dataset:')
    print(f'- Using {args.api} API')

    # Generate dataset and print summary
    alternative_df = generator.create_alternative_dataset()
    print_generation_summary(alternative_df)


if __name__ == '__main__':
    main()

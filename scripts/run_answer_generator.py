import argparse
from textwrap import dedent
from pathlib import Path
from mmlu_eval.model_api import ClaudeAPI, DeepseekAPI, ClaudeConfig, DeepseekConfig
from mmlu_eval.answer_generator import DatasetGenerator, AlternativeAnswerConfig
from mmlu_eval.paths import (
    get_ref_data_path, 
    ALTERNATIVE_DATA_DIR,
    MMLU_TEST_FILE
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate alternative answers for MMLU questions using LLMs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent('''
            Example usage:
            python generate_model_answers.py \\
                --df-path ref_dataframes/mmlu_test.parquet \\
                --output-dir alternative_datasets \\
                --num-samples 1000 \\
                --api claude \\
                --seed 42
        ''')
    )
    
    parser.add_argument(
        '--df-path',
        type=str,
        required=True,
        help='Path to original MMLU dataset parquet file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save generated datasets'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1000,
        help='Number of questions to sample from original dataset'
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
        '--difficulty',
        type=str,
        default="significantly difficult but still clearly incorrect",
        help='Description of desired difficulty for generated answers'
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize API
    if args.api == 'claude':
        api = ClaudeAPI(ClaudeConfig())
    else:
        api = DeepseekAPI(DeepseekConfig())
    
    # Configure generator
    config = AlternativeAnswerConfig(
        num_samples=args.num_samples,
        difficulty_requirement=args.difficulty
    )
    
    # Initialize generator
    generator = DatasetGenerator(
        df_path=get_ref_data_path(MMLU_TEST_FILE),
        api=api,
        config=config,
        output_path=ALTERNATIVE_DATA_DIR
    )
    
    # Generate dataset
    print(f"\nGenerating alternative dataset:")
    print(f"- Sampling {args.num_samples} questions")
    print(f"- Using {args.api} API")
    print(f"- Output directory: {ALTERNATIVE_DATA_DIR}")
    print(f"- Seed: {args.seed}\n")
    
    alternative_df = generator.create_alternative_dataset(seed=args.seed)
    
    # Print summary
    print(f"\nGeneration complete:")
    print(f"- Generated {len(alternative_df)} questions")
    if 'subject' in alternative_df.columns:
        subjects = alternative_df['subject'].value_counts()
        print("\nSubject distribution:")
        for subject, count in subjects.items():
            print(f"- {subject}: {count}")
    
    # Print output files
    model_id = api.config.model if hasattr(api, 'config') else "unknown_model"
    output_file = ALTERNATIVE_DATA_DIR / f"alternative_dataset_{model_id}_{args.seed}.parquet"
    config_file = ALTERNATIVE_DATA_DIR / f"alternative_dataset_{model_id}_{args.seed}_config.json"
    indices_file = ALTERNATIVE_DATA_DIR / f"sampled_indices_{args.seed}.json"
    print(f"\nOutput files:")
    print(f"- Dataset: {output_file}")
    print(f"- Config:  {config_file}")

if __name__ == "__main__":
    main()

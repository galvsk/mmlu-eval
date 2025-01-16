import numpy as np
import pandas as pd
import random
from typing import List, Literal, Tuple
from dataclasses import dataclass
import json
from pathlib import Path
from mmlu_eval.model_api import ModelAPI, ClaudeAPI, DeepseekAPI, ClaudeConfig, DeepseekConfig
from mmlu_eval.analysis import MMLU_CATEGORY_MAP
from mmlu_eval.paths import MMLU_TEST_FILE, ALTERNATIVE_DATA_DIR


@dataclass
class AlternativeAnswerConfig:
    """Configuration for generating alternative answers"""
    num_samples: int = 2000
    difficulty_requirement: str = "significantly difficult but still clearly incorrect"
    system_context: str = """You are an expert test question writer.
    Your task is to generate alternative wrong answers that are challenging and plausible."""

class DatasetGenerator:
    def __init__(
        self,
        df_path: str = MMLU_TEST_FILE,
        api: Literal['claude', 'deepseek'] = 'claude',
        save_frequency: int = 100,
        seed: int = 666
    ):
        """Initialize the dataset generator.

        Args:
            df_path: Path to original MMLU dataset
            api: Model API instance for generating answers
            output_path: Where to save the generated dataset
            save_frequency: How often to save progress (in number of questions)
        """
        self.api_str = api
        self.df_path = Path(df_path)
        self.config = AlternativeAnswerConfig()
        self.save_frequency = save_frequency
        self.seed = seed

        # Load and process base dataframe
        self.df = pd.read_parquet(df_path)
        # Remap subjects to coarser grained values
        assert len(np.unique(self.df.subject)) == len(MMLU_CATEGORY_MAP)
        self.df['subject'] = self.df['subject'].map(MMLU_CATEGORY_MAP)
        # Only consider subset of subjects
        self.df = self.df[self.df.subject.isin(['STEM', 'Law & Ethics', 'Social Sciences'])]

        if self.api_str == 'claude':
            self.api = ClaudeAPI(ClaudeConfig())
        elif self.api_str == 'deepseek':
            self.api = DeepseekAPI(DeepseekConfig())
        else:
            raise ValueError(f"Unsupported API type: {api}")

    def _get_output_paths(self) -> Tuple[Path, Path]:
        """Get paths for output files."""
        output_file = ALTERNATIVE_DATA_DIR / f"{self.api_str}_generated_dataframe.parquet"
        config_file = ALTERNATIVE_DATA_DIR / f"{self.api_str}_generated_config.json"
        return output_file, config_file

    def sample_questions(self) -> pd.DataFrame:
        """Create a stratified sample of questions."""
        random.seed(self.seed)
        sampled = self.df.sample(n=self.config.num_samples, random_state=self.seed)
        return sampled.reset_index(drop=True)

    def generate_alternative_answers(self, question: str, correct_answer: str,
                                   wrong_answers: List[str]) -> List[str]:
        """Generate alternative wrong answers using the model."""
        prompt = (
            f'Given this multiple choice question and its answers, generate 3 new alternative wrong answers. '
            f'The new wrong answers should be {self.config.difficulty_requirement}.\n\n'
            f'Question: {question}\n\n'
            f'Correct answer: {correct_answer}\n\n'
            f'Current wrong answers: {", ".join(wrong_answers)}\n\n'
            'Provide exactly 3 new wrong answers, one per line, no labels or prefixes. Each answer should be relevant '
            'to the question domain and challenging but clearly incorrect to a knowledgeable person.'
        )

        response = self.api.get_completion(prompt)
        alternatives = [line.strip() for line in response['prediction'].split('\n')
                      if line.strip()][:3]
        return alternatives

    def create_alternative_dataset(self) -> pd.DataFrame:
        """Create new dataset with alternative wrong answers."""
        output_file, config_file = self._get_output_paths()

        # Try to load existing dataset or create new one
        if output_file.exists():
            df = pd.read_parquet(output_file)
            print(f"\nLoading existing dataset with {len(df)} questions")
        else:
            # Sample questions and initialize DataFrame
            sampled_df = self.sample_questions()
            df = pd.DataFrame({
                'question': sampled_df['question'],
                'correct_answer': sampled_df.apply(lambda x: x.choices[x.answer], axis=1),
                'original_wrong_answers': sampled_df.apply(
                    lambda x: [c for i, c in enumerate(x.choices) if i != x.answer],
                    axis=1
                ),
                'generated_wrong_answers': pd.NA,
                'subject': sampled_df['subject']
            })
            print(f"\nInitialized new dataset with {len(df)} questions")

        # Process only questions with NaN generated_wrong_answers
        remaining_mask = df['generated_wrong_answers'].isna()
        remaining_df = df[remaining_mask]
        print(f"Generating alternatives for {len(remaining_df)} remaining questions...")

        for idx, row in remaining_df.iterrows():
            try:
                alternatives = self.generate_alternative_answers(
                    row.question,
                    row.correct_answer,
                    row.original_wrong_answers
                )
                df.loc[idx, 'generated_wrong_answers'] = alternatives

                # Save progress periodically
                completed = (~df['generated_wrong_answers'].isna()).sum()
                if completed % self.save_frequency == 0:
                    df.to_parquet(output_file)
                    print(f"Progress: {completed}/{len(df)} questions")

            except Exception as e:
                print(f"Error generating alternatives for question {idx}: {e}")
                continue

        # Save final dataset and config
        df.to_parquet(output_file, index=False)


        config = {
            'original_dataset': str(self.df_path),
            'num_samples': self.config.num_samples,
            'seed': self.seed,
            'model': self.api_str,
            'difficulty_requirement': self.config.difficulty_requirement,
            'total_questions_generated': (~df['generated_wrong_answers'].isna()).sum()
        }
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        completed = (~df['generated_wrong_answers'].isna()).sum()
        print(f"\nGeneration complete!")
        print(f"- Generated answers for {completed}/{len(df)} questions")
        print(f"- Results saved to {output_file}")

        return df


if __name__ == "__main__":
    import random

    df_path = MMLU_TEST_FILE
    df = pd.read_parquet(df_path)
    # Select random question
    random_idx = random.randint(0, len(df) - 1)
    test_row = df.iloc[random_idx]
    test_question = test_row.question
    test_choices = test_row.choices
    correct_answer_idx = test_row.answer

    print(f"Selected random question (index {random_idx}):")
    if 'subject' in test_row:
        print(f"Subject: {test_row.subject}")

    # Create generators (using actual df_path now)
    claude_gen = DatasetGenerator(
        df_path=df_path,
        api='claude',
    )

    deepseek_gen = DatasetGenerator(
        df_path=df_path,
        api='deepseek',
    )

    # Generate alternative answers with both models
    correct_answer = test_choices[correct_answer_idx]
    wrong_answers = [c for i, c in enumerate(test_choices) if i != correct_answer_idx]

    print(f"\nQ: {test_question}")
    print(f"\nCorrect Answer: {correct_answer}")
    print(f"\nOriginal Wrong Answers:")
    for i, ans in enumerate(wrong_answers, 1):
        print(f"{i}. {ans}")

    print("\nGenerating alternatives with Claude...")
    claude_alternatives = claude_gen.generate_alternative_answers(
        test_question, correct_answer, wrong_answers
    )

    print("\nGenerating alternatives with Deepseek...")
    deepseek_alternatives = deepseek_gen.generate_alternative_answers(
        test_question, correct_answer, wrong_answers
    )

    # Print results
    print("\nClaude Alternative Wrong Answers:")
    for i, alt in enumerate(claude_alternatives, 1):
        print(f"{i}. {alt}")

    print("\nDeepseek Alternative Wrong Answers:")
    for i, alt in enumerate(deepseek_alternatives, 1):
        print(f"{i}. {alt}")

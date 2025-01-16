import numpy as np
import pandas as pd
import random
from typing import List, Dict
from dataclasses import dataclass
import json
from pathlib import Path
from model_api import ModelAPI, ClaudeAPI, DeepseekAPI, ClaudeConfig, DeepseekConfig
from analysis.utils import MMLU_CATEGORY_MAP

@dataclass
class AlternativeAnswerConfig:
    """Configuration for generating alternative answers"""
    num_samples: int = 1200
    difficulty_requirement: str = "significantly difficult but still clearly incorrect"
    system_context: str = """You are an expert test question writer. 
    Your task is to generate alternative wrong answers that are challenging and plausible."""

class DatasetGenerator:
    def __init__(
        self,
        df_path: str,
        api: ModelAPI,
        config: AlternativeAnswerConfig,
        output_path: str
    ):
        """Initialize the dataset generator.
        
        Args:
            df_path: Path to original MMLU dataset
            api: Model API instance for generating answers
            config: Configuration for alternative answer generation
            output_path: Where to save the generated dataset
        """
        self.df_path = Path(df_path)
        self.api = api
        self.config = config
        self.output_path = Path(output_path)
        self.df = pd.read_parquet(df_path)
        # Remap subjects to coarser grained values
        assert len(np.unique(self.df.subject)) == len(MMLU_CATEGORY_MAP)
        self.df['subject'] = self.df['subject'].map(MMLU_CATEGORY_MAP)
        # Only consider subset
        self.df = self.df[self.df.subject.isin(['STEM', 'Law & Ethics', 'Social Sciences'])]

        
        # Get model identifier for filenames
        if hasattr(api, 'config'):
            self.model_id = f"{api.config.model}"
        else:
            raise ValueError(f"Unsupported API type: {api}")
        
    def sample_questions(self, seed: int = 666) -> pd.DataFrame:
        """Create a stratified sample of questions.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with sampled questions
        """
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Stratify by subject if available
        if 'subject' in self.df.columns:
            # Calculate samples per subject proportionally
            sampled = self.df.groupby('subject', group_keys=False).apply(
                lambda x: x.sample(
                    n=min(
                        len(x), 
                        max(1, int(self.config.num_samples * len(x) / len(self.df)))
                    ),
                    random_state=seed  # Use same seed for each group
                )
            )
            
            # Adjust if we got too many/few questions
            if len(sampled) > self.config.num_samples:
                sampled = sampled.sample(n=self.config.num_samples, random_state=seed)
            elif len(sampled) < self.config.num_samples:
                remaining = self.config.num_samples - len(sampled)
                additional = self.df.drop(sampled.index).sample(
                    n=remaining, 
                    random_state=seed
                )
                sampled = pd.concat([sampled, additional])
        else:
            sampled = self.df.sample(n=self.config.num_samples, random_state=seed)
            
        return sampled.sort_index()  # Sort for consistent ordering
    
    def generate_alternative_answers(self, question: str, correct_answer: str, 
                                   wrong_answers: List[str]) -> List[str]:
        """Generate alternative wrong answers using the model."""
        prompt = f"""Given this multiple choice question and its answers, generate 3 new alternative wrong answers.
        The new wrong answers should be {self.config.difficulty_requirement}.

        Question: {question}
        
        Correct answer: {correct_answer}
        
        Current wrong answers: 
        {', '.join(wrong_answers)}
        
        Provide exactly 3 new wrong answers, one per line, no labels or prefixes. Each answer should be relevant 
        to the question domain and challenging but clearly incorrect to a knowledgeable person."""
        
        response = self.api.get_completion(prompt)
        
        # Split response into lines and take first 3 non-empty lines
        alternatives = [line.strip() for line in response['prediction'].split('\n') 
                      if line.strip()][:3]
        
        return alternatives
    
    def create_alternative_dataset(self, seed: int = 42) -> pd.DataFrame:
        """Create new dataset with alternative wrong answers.
        
        The dataset format allows for flexible use of original and new wrong answers:
        - 'question': Original question
        - 'correct_answer': The correct answer
        - 'original_wrong_answers': List of original wrong answers
        - 'generated_wrong_answers': List of model-generated wrong answers
        - 'subject': Subject category (if available)
        """
        # Sample questions
        sampled_df = self.sample_questions(seed)
        
        # Generate alternative answers for each question
        new_data = []
        for idx, row in sampled_df.iterrows():
            correct_answer = row.choices[row.answer]
            wrong_answers = [c for i, c in enumerate(row.choices) if i != row.answer]
            
            try:
                alternatives = self.generate_alternative_answers(
                    row.question, correct_answer, wrong_answers
                )
                
                # Store all answers separately for maximum flexibility
                new_data.append({
                    'question': row.question,
                    'correct_answer': correct_answer,
                    'original_wrong_answers': wrong_answers,
                    'generated_wrong_answers': alternatives,
                    'subject': row.get('subject', 'unknown'),
                    'original_idx': idx  # Store original index for reference
                })
            except Exception as e:
                print(f"Error generating alternatives for question {idx}: {e}")
                continue
        
        new_df = pd.DataFrame(new_data)
        
        # Save the dataset with model identifier
        output_file = self.output_path / f"alternative_dataset_{self.model_id}.parquet"
        new_df.to_parquet(output_file)
        
        # Save configuration
        config_file = self.output_path / f"alternative_dataset_{self.model_id}_config.json"
        config = {
            'original_dataset': str(self.df_path),
            'num_samples': self.config.num_samples,
            'seed': seed,
            'model': self.model_id,
            'difficulty_requirement': self.config.difficulty_requirement
        }
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
            
        # Also save the sampled indices for reproducibility
        indices_file = self.output_path / f"sampled_indices_{seed}.json"
        if not indices_file.exists():  # Only save once per seed
            with open(indices_file, 'w') as f:
                json.dump({
                    'seed': seed,
                    'indices': sorted(sampled_df.index.tolist())
                }, f, indent=2)
        
        return new_df


if __name__ == "__main__":
    import random
    
    # Load real MMLU test data
    df_path = "ref_dataframes/mmlu_test.parquet"
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
    
    # Initialize both APIs
    claude_api = ClaudeAPI(ClaudeConfig())
    deepseek_api = DeepseekAPI(DeepseekConfig())
    
    # Common configuration
    config = AlternativeAnswerConfig(
        num_samples=1000,
        difficulty_requirement="significantly difficult but still clearly incorrect"
    )
    
    # Create generators (using actual df_path now)
    claude_gen = DatasetGenerator(
        df_path=df_path,
        api=claude_api,
        config=config,
        output_path="."
    )
    
    deepseek_gen = DatasetGenerator(
        df_path=df_path,
        api=deepseek_api,
        config=config,
        output_path="."
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

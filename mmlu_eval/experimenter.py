import os
import re
import numpy as np
import pandas as pd
import json
import time
from typing import List, Optional, Dict, Literal
from pathlib import Path
import hashlib
from mmlu_eval.formatter import MMLUPromptDefault, MMLUPromptAlternative
from mmlu_eval.model_api import ClaudeAPI, DeepseekAPI, ClaudeConfig, DeepseekConfig, ModelAPI


class MMLUExperimenter:
    def __init__(
        self,
        experiment_path: str,
        df_path: str,
        description: None | str = None,
        api: Literal['claude', 'deepseek'] = 'claude',
        save_frequency: int = 10,
        prompt_style: MMLUPromptDefault = MMLUPromptDefault
    ):
        """
        Initialize or resume an MMLU experiment.
        
        Args:
            experiment_path: Path to experiment directory
            df_path: Path to dataset parquet
            description: Description of experiment (required for new experiments)
            api : Which LLM to use (Claude and Deepseek supported)
            save_frequency: How often to save results (in number of questions)
            prompt_style: Class defining how to present the MMLU questions
        """
        self.experiment_path = Path(experiment_path)
        self.df_path = Path(df_path)
        self.save_frequency = save_frequency
        
        # Set up client
        if api == 'claude':
            self.api = ClaudeAPI(ClaudeConfig())
        elif api == 'deepseek':
            self.api = DeepseekAPI(DeepseekConfig())
        else:
            raise ValueError(f"Unsupported API type: {api}")
        
        # Load data
        self.df = self._load_and_validate_data()
        
        # Set up experiment
        self.results_path = Path(os.path.join(self.experiment_path,  "results.parquet"))
        self.config_path = Path(os.path.join(self.experiment_path, "config.json"))
        
        # Add prompt style for posing questions
        self.prompt_style = prompt_style
       
        if self._is_existing_experiment():
            self._resume_experiment()
        else:
            if description is None:
                raise ValueError("Description required for new experiments")
            self._initialize_experiment(description)
    
    def _load_and_validate_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load and validate the MMLU datasets."""
        if not self.df_path.exists():
            raise FileNotFoundError(f"Dataframe not found: {self.df_path}")
        
        df = pd.read_parquet(self.df_path)
        return df
    
    def _is_existing_experiment(self) -> bool:
        """Check if this is an existing experiment."""
        return self.experiment_path.exists()
    
    def _initialize_experiment(self, description: str):
        """Set up a new experiment."""
        # Create experiment directory
        self.experiment_path.mkdir(parents=True, exist_ok=False)

        # Get model info from API config
        model_info = {
            'type': 'claude' if isinstance(self.api, ClaudeAPI) else 'deepseek',
            'model': self.api.config.model
        }
        
        # Save experiment config
        config = {
            'description': description,
            'model': model_info,
            'save_frequency': self.save_frequency,
            'df_path': str(self.df_path),
            'data_hash': hashlib.md5(self.df.to_string().encode()).hexdigest(),
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'last_updated': None
        }
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Initialize experiment results DataFrame
        self.exp_df = self.df.copy(deep=True)
        self.exp_df['predicted'] = pd.NA
        
        # Save initial state
        self._save_results()
        print(f"Initialized new experiment: {description}")
    
    def _resume_experiment(self):
        """Resume an existing experiment."""
        # Load config
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load experiment config: {e}")
        
        # Validate data consistency
        current_hash = hashlib.md5(self.df.to_string().encode()).hexdigest()
        if current_hash != self.config['data_hash']:
            raise ValueError("Dataframe has changed since experiment creation")
        
        # Load results
        total_predictions = 0
        if self.results_path.exists():
            saved_df = pd.read_parquet(self.results_path)
            # Update only rows that have predictions
            mask = saved_df['predicted'].notna()
            self.exp_df = self.df.copy(deep=True)
            self.exp_df['predicted'] = saved_df['predicted']
            total_predictions += mask.sum()
        
        print(f"Resumed experiment: {self.config['description']}")
        print(f"Loaded {total_predictions} existing predictions")
    
    def _save_results(self):
        """Save current results and update config."""
        # Save results
        self.exp_df.to_parquet(self.results_path, index=False)
        
        # Calculate and print progress
        total = len(self.exp_df)
        completed = self.exp_df['predicted'].notna().sum()
        
        print(f"\nProgress: {completed} / {total} questions ({(completed / total * 100) :.2f}%)\n")
        
        # Update config
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        config['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def _format_prompt(self, question: str, choices: List[str], answer: int) -> tuple[str, int, Dict[int, int]]:
        """Format the question and choices into a prompt for Claude.
        Returns:
            tuple: (formatted_prompt, answer_index, position_mapping)
        """
        formatter = self.prompt_style(
            question=question,
            choices=choices,
            answer=answer
        )
        return formatter.format_question()
    
    def _parse_response(self, response: str, position_mapping: Dict[int, int]) -> Optional[int]:
        """Parse LLMs response to extract the predicted answer index.
        Maps the response back to the original answer position.
        """
        if len(position_mapping) == 4:
            reference_answers = ('A', 'B', 'C', 'D')
        elif len(position_mapping) == 7:
            reference_answers = ('A', 'B', 'C', 'D', 'E', 'F', 'G')
        else:
            raise ValueError('Only 4 or 7 multiple choice answers valid.')

        try:
            # Valid answers can be of the form A-G, upper or lower case, with a '.' or ')' added after letter
            pattern = r'^([A-Ga-g])[.).]?.*$'
            match = re.match(pattern, response.strip())
            # If we find a match, extract first letter and convert to index in answer list
            if match:
                letter = match.group(1).upper()
                response_idx = ord(letter) - ord('A')
                return position_mapping[response_idx]
            else:
                print(f"Could not parse response : {response}")
                return -1
        except Exception:
            return None

    def run_experiment(
        self,
        max_questions: None | int = None,
        retry_errors: bool = True,
        retry_delay: int = 5
    ):
        """
        Run the experiment on specified number of questions.
        
        Args)
            max_questions: Maximum number of questions to process
            retry_errors: Whether to retry failed API calls
            retry_delay: Seconds to wait between retries
        """
        # Get unanswered questions
        unanswered = self.exp_df[self.exp_df['predicted'].isna()].index
        if max_questions:
            unanswered = unanswered[:max_questions]
            
        print(f"Processing {len(unanswered)} questions...")
        
        for idx, row_idx in enumerate(unanswered):
            row = self.exp_df.loc[row_idx]
            prompt, _, position_mapping = self._format_prompt(
                row['question'], 
                row['choices'], 
                row['answer']
            ) 
            success = False
            while not success:
                try:
                    response = self.api.get_completion(prompt)
                    # Format answer as index from choices (if possible)
                    output = self._parse_response(response['prediction'], position_mapping)
                    if output is not None:
                        # Save predicted answer
                        self.exp_df.loc[row_idx, 'predicted'] = output
                        success = True
                    else:
                        print(f"Could not parse prediction from response: {response}")
                        if not retry_errors:
                            break
                            
                except Exception as e:
                    print(f"Error processing question {row_idx}: {e}")
                    if not retry_errors:
                        break
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
            
            # Save checkpoint periodically (and print accuracy)
            if (idx + 1) % self.save_frequency == 0:
                self._save_results()
                acc = self.get_accuracy()
                print(f"Saved checkpoint after {idx + 1} questions, accuracy: {acc * 100:.2f}%")
                
        # Final save
        self._save_results()
        acc = self.get_accuracy()
        print(f"\nFinal accuracy: {acc * 100:.2f}%")
    
    def get_accuracy(self) -> float:
        """Calculate accuracy for currently answered questions"""
        mask = self.exp_df['predicted'].notna()
        if mask.sum() == 0:
            return 0.0
        return (self.exp_df[mask]['predicted'] == self.exp_df[mask]['answer']).mean()
    
    def get_results_summary(self) -> dict:
        """Get summary of experiment results."""
        with open(self.config_path, 'r') as f:
            config = json.load(f)
            
        results = {
            'description': config['description'],
            'model': config['model'],
            'created_at': config['created_at'],
            'last_updated': config['last_updated'],
            'questions': len(self.exp_df),
            'completed_questions': self.exp_df['predicted'].notna().sum(),
            'total_questions': len(self.exp_df),
            'accuracy': self.get_accuracy(),
        }
        return results

class AlternativeExperimenter(MMLUExperimenter):
    """Experimenter for evaluating models on alternative answer sets."""

    def __init__(
        self,
        *args,
        eval_mode: Literal['generated_only', 'all_answers'] = 'generated_only',
        **kwargs
    ):
        """
        Initialize experimenter with specified evaluation mode.

        Args:
            eval_mode: Whether to evaluate on just generated answers ('generated_only')
                      or all answers ('all_answers')
            *args, **kwargs: Arguments passed to parent MMLUExperimenter
        """
        self.eval_mode = eval_mode
        super().__init__(*args, prompt_style=MMLUPromptAlternative, **kwargs)

    def _load_and_validate_data(self) -> pd.DataFrame:
        """Load alternative dataset and convert to standard MMLU format."""
        if not self.df_path.exists():
            raise FileNotFoundError(f"Dataframe not found: {self.df_path}")

        df = pd.read_parquet(self.df_path)

        # Validate required columns
        required_cols = ['question', 'correct_answer', 'original_wrong_answers',
                        'generated_wrong_answers']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Validate data types
        if not isinstance(df['generated_wrong_answers'].iloc[0], np.ndarray):
            raise ValueError("generated_wrong_answers should be a numpy array")
        if not isinstance(df['original_wrong_answers'].iloc[0], np.ndarray):
            raise ValueError("original_wrong_answers should be a numpy array")

        # Convert to standard MMLU format
        formatted_df = pd.DataFrame()
        formatted_df['question'] = df['question']

        # Combine answers based on eval_mode
        if self.eval_mode == 'generated_only':
            formatted_df['choices'] = df.apply(
                lambda row: [row['correct_answer']] + row['generated_wrong_answers'].tolist(),
                axis=1
            )
        else:  # all_answers
            formatted_df['choices'] = df.apply(
                lambda row: (
                    [row['correct_answer']] +
                    row['original_wrong_answers'].tolist() +
                    row['generated_wrong_answers'].tolist()
                ),
                axis=1
            )

        # Correct answer is first in choices list
        formatted_df['answer'] = 0

        # Keep subject if present
        if 'subject' in df.columns:
            formatted_df['subject'] = df['subject']

        return formatted_df

    def _format_prompt(self, question: str, choices: List[str], answer: int) -> tuple[str, int, Dict[int, int]]:
        """Format the question and choices into a prompt for Claude.
        Returns:
            tuple: (formatted_prompt, answer_index, position_mapping)
        """
        formatter = MMLUPromptAlternative(
            question=question,
            choices=choices,
            answer=answer
        )
        return formatter.format_question()

    def get_results_summary(self) -> dict:
        """Add evaluation mode to results summary."""
        summary = super().get_results_summary()
        summary['eval_mode'] = self.eval_mode
        return summary

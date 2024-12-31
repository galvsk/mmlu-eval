import os
import re
import pandas as pd
import anthropic
import json
import time
from typing import List, Optional, Dict
from pathlib import Path
import hashlib
from mmlu_formatter import  MMLUPromptDefault
from utils import get_api_key

class MMLUExperimenter:
    def __init__(
        self,
        experiment_path: str,
        df_path: str,
        description: None | str = None,
        model: str = "claude-3-sonnet-20240229",
        save_frequency: int = 10,
        prompt_style: MMLUPromptDefault = MMLUPromptDefault
    ):
        """
        Initialize or resume an MMLU experiment.
        
        Args:
            experiment_path: Path to experiment directory
            df_path: Path to dataset parquet
            description: Description of experiment (required for new experiments)
            model: Claude model to use
            save_frequency: How often to save results (in number of questions)
            prompt_style: Class defining how to present the MMLU questions
        """
        self.experiment_path = Path(experiment_path)
        self.df_path = Path(df_path)
        self.model = model
        self.save_frequency = save_frequency
        
        # Set up client
        self.client = anthropic.Client(api_key=get_api_key())
        
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
        
        # Save experiment config
        config = {
            'description': description,
            'model': self.model,
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
        self.exp_df.to_parquet(self.results_path)
        
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

    def _format_prompt(self, question: str, choices: List[str], answer: int) -> Tuple[str, int, Dict[int, int]]:
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
        """Parse Claude's response to extract the predicted answer index.
        Maps the response back to the original answer position.
        """
        try:
            output = response[0].text
            if len(output) == 1 and output in ('A', 'B', 'C', 'D'):
                # Convert letter to number (A=0, B=1, C=2, D=3)
                letter = output[0].upper()
                response_idx = ord(letter) - ord('A')
                # Map back to original position
                return position_mapping[response_idx]
            else:
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
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=5,
                        temperature=0,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    # Extract prediction from response
                    prediction = self._parse_response(response.content, position_mapping)
                    if prediction is not None:
                        self.exp_df.loc[row_idx, 'predicted'] = prediction
                        success = True
                    else:
                        print(f"Could not parse prediction from response: {response.content}")
                        if not retry_errors:
                            break
                            
                except Exception as e:
                    print(f"Error processing question {row_idx}: {e}")
                    if not retry_errors:
                        break
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
            
            # Save checkpoint periodically
            if (idx + 1) % self.save_frequency == 0:
                self._save_results()
                print(f"Saved checkpoint after {idx + 1} questions")
                
        # Final save
        self._save_results()
    
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

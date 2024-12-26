import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict

class MMLUData:
    def __init__(self, data_dir: str = '.'):
        self.data_dir = Path(data_dir)
        self.data = pd.DataFrame({})
        
    def gather_and_format(self) -> None:
        """Load all parquet files from directory structure."""
        parquet_files = list(self.data_dir.rglob("*parquet"))
        all_data = []
        print(f"Found {len(parquet_files)} files")
        
        for file in parquet_files:
            try:
                df = pd.read_parquet(file)
                # Only load train and test folds
                if 'test' in file.stem:
                    split = 'test'
                elif 'train' in file.stem:
                    split = 'train'
                else:
                    continue
                
                # Label train and a test
                df['fold'] = split
                records = df.to_dict('records')
                all_data.extend(records)
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue
        self.data = pd.DataFrame(all_data)
    
    def summarize(self) -> None:
        """Print summary of loaded data."""
        if self.data.empty:
            print("No data loaded. Run gather_and_format() first.")
            return
        
        print(f"\nTotal questions: {len(self.data)}")
        print(f"\nColumns present: {self.data.columns.tolist()}")
        print(f"\nTraining vs Test ;\n{self.data.fold.value_counts()}")
        nan_columns = self.data.columns[self.data.isna().any()].tolist()
        assert len(nan_columns) == 0, f"Columns with NaNs: {nan_columns}"        
        
    def sample_question(self) -> Dict:
        """Return a random sample question with formatted output."""
        if self.data.empty:
            print("No data loaded. Run gather_data() first.")
            return {}
        question = self.data.sample(n=1).iloc[0].to_dict()
        
        print("\nSample question:")
        for k, v in question.items():
            print(f"{k}: {v}")
        
        return question
    
    def save_formatted(self, save_path: str = 'ref_dataframes') -> None:
        if self.data.empty:
            print("No data loaded. Run gather_and_format() first.")
            return
        
        os.makedirs(save_path, exist_ok=True)
        # Split the dataset up and save separately
        trn_df = self.data[self.data.fold == 'train'].reset_index()
        test_df = self.data[self.data.fold == 'test'].reset_index()
        trn_df.to_parquet(os.path.join(save_path, 'mmlu_train.parquet'))
        test_df.to_parquet(os.path.join(save_path, 'mmlu_test.parquet'))
    

if __name__ == "__main__":
    mmlu = MMLUData("MMLU")
    mmlu.gather_and_format()
    mmlu.summarize()
    question = mmlu.sample_question()
    print("Saving formatted dataframes.")
    mmlu.save_formatted()

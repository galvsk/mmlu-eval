from dataclasses import dataclass
from typing import Literal, Dict, Any, Optional
import pandas as pd
import anthropic
from openai import OpenAI
from mmlu_formatter import MMLUPromptDefault
from utils import get_api_key

@dataclass
class APIConfig:
    """Configuration for API clients."""
    client_type: Literal['claude', 'deepseek']
    model: str
    base_url: str | None = None
    system_prompt: str | None = None

API_CONFIGS = {
    'claude': APIConfig(
        client_type='claude',
        model="claude-3-5-sonnet-20241022"
    ),
    'deepseek': APIConfig(
        client_type='deepseek',
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        system_prompt="You are a helpful assistant"
    )
}

class LLMTester:
    """Class to handle LLM API testing."""
    
    def __init__(self):
        self.api_keys = {
            'claude': get_api_key('claude'),
            'deepseek': get_api_key('deepseek')
        }
        self._clients = {}
        self.letters = ('A', 'B', 'C', 'D')

    def _get_client(self, client_type: str) -> Any:
        """Get or create API client."""
        if client_type not in self._clients:
            if client_type == 'claude':
                self._clients[client_type] = anthropic.Anthropic(api_key=self.api_keys['claude'])
            elif client_type == 'deepseek':
                self._clients[client_type] = OpenAI(
                    api_key=self.api_keys['deepseek'],
                    base_url=API_CONFIGS[client_type].base_url
                )
        return self._clients[client_type]

    def _call_claude_api(self, prompt: str) -> str:
        """Make API call to Claude."""
        client = self._get_client('claude')
        response = client.messages.create(
            model=API_CONFIGS['claude'].model,
            max_tokens=5,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text[0]

    def _call_deepseek_api(self, prompt: str) -> str:
        """Make API call to Deepseek."""
        client = self._get_client('deepseek')
        response = client.chat.completions.create(
            model=API_CONFIGS['deepseek'].model,
            messages=[
                {"role": "system", "content": API_CONFIGS['deepseek'].system_prompt},
                {"role": "user", "content": prompt},
            ],
            logprobs=True,
            max_tokens=5,
            top_p=1,
            temperature=0,
            stream=False
        )
        return response.choices[0].message.content

    def test_single_question(self, prompt: Dict[str, Any], client_type: str = 'claude') -> Dict[str, Any]:
        """Test a single MMLU question using specified API."""
        if client_type not in API_CONFIGS:
            raise ValueError(f"Unsupported client type: {client_type}. "
                           f"Supported types are: {list(API_CONFIGS.keys())}")

        # Format the prompt
        prompt_formatter = MMLUPromptDefault(
            question=prompt['question'],
            choices=prompt['choices'],
            answer=prompt['answer']
        )
        formatted_question, correct_answer, position_mapping = prompt_formatter.format_question()

        # Get prediction from appropriate API
        if client_type == 'claude':
            predicted = self._call_claude_api(formatted_question)
        else:  # deepseek
            predicted = self._call_deepseek_api(formatted_question)

        return {
            'predicted': predicted,
            'correct': chr(65 + correct_answer),  # Convert to letter (A, B, C, D)
            'is_correct': predicted == chr(65 + correct_answer),
            'full_prompt': formatted_question
        }


def main():
    """Main execution function comparing Claude and Deepseek responses."""
    tester = LLMTester()
    df = pd.read_parquet('ref_dataframes/mmlu_test.parquet')
    question = df.loc[0]
    results = {}
    for client_type in API_CONFIGS:
        output = tester.test_single_question(question, client_type)
        results[client_type] = output
    
    print(f"{results['claude']['full_prompt']} \n"
          f"Claude, answer : {results['claude']['predicted']}, correct : {results['claude']['is_correct']}, \n"
          f"Deepseek, answer : {results['deepseek']['predicted']}, correct : {results['deepseek']['is_correct']}")

if __name__ == "__main__":
    main()

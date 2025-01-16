import random
import pandas as pd
from mmlu_eval.formatter import MMLUPromptDefault
from mmlu_eval.model_api import ClaudeAPI, DeepseekAPI, ClaudeConfig, DeepseekConfig
from mmlu_eval.paths import get_ref_data_path, MMLU_TEST_FILE


class LLMTester:
    def __init__(self):
        self.claude_api = ClaudeAPI(ClaudeConfig())
        self.deepseek_api = DeepseekAPI(DeepseekConfig())

    def test_single_question(self, prompt: dict, client_type: str = 'claude') -> dict:
        # Format the prompt
        prompt_formatter = MMLUPromptDefault(
            question=prompt['question'],
            choices=prompt['choices'],
            answer=prompt['answer']
        )
        formatted_question, correct_answer, position_mapping = prompt_formatter.format_question()

        # Get prediction from appropriate API
        if client_type == 'claude':
            response = self.claude_api.get_completion(formatted_question)
        elif client_type == 'deepseek':
            response = self.deepseek_api.get_completion(formatted_question)
        else:
            raise ValueError(f"Unsupported client type: {client_type}")

        return {
            'predicted': response['prediction'],
            'correct': chr(65 + correct_answer),
            'is_correct': response['prediction'] == chr(65 + correct_answer),
            'full_prompt': formatted_question,
        }


def main():
    tester = LLMTester()
    # Load test data using paths module
    df = pd.read_parquet(get_ref_data_path(MMLU_TEST_FILE))
    idx = random.randint(0, len(df)-1)
    question = df.iloc[idx].to_dict()

    print(f"\nSelected question {idx}")
    if 'subject' in question:
        print(f"Subject: {question['subject']}")

    # Test with both APIs
    claude_result = tester.test_single_question(question, 'claude')
    deepseek_result = tester.test_single_question(question, 'deepseek')

    # Print results
    print(f"\nQuestion asked:\n{claude_result['full_prompt']}\n")

    print(f"Claude:")
    print(f"Answer: {claude_result['predicted']}")
    print(f"Correct: {claude_result['is_correct']}\n")

    print(f"Deepseek:")
    print(f"Answer: {deepseek_result['predicted']}")
    print(f"Correct: {deepseek_result['is_correct']}")


if __name__ == "__main__":
    main()

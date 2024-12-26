import anthropic
from generate_dataframes import MMLUData
from utils import get_api_key, MMLUPromptDefault


def test_single_question(api_key, prompt):
    """Test a single MMLU question using Claude."""
    client = anthropic.Anthropic(api_key=api_key)
    input_prompt = MMLUPromptDefault(question=prompt['question'], choices=prompt['choices']).format()
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=5,
        temperature=0,
        messages=[{"role": "user", "content": input_prompt}]
    )
    letters = ('A', 'B', 'C', 'D')
    # Extract predicted answer from content list
    predicted = response.content[0].text[0]
    correct = chr(ord(predicted))
    print(f"\nQuestion: {prompt['question']}")
    print("\nChoices:")
    for i, choice in enumerate(prompt['choices']):
        print(f"{letters[i]}) {choice}")
    print(f"\nClaude's answer: {predicted}")
    print(f"Correct answer: {correct}")
    print(f"Correct? {predicted == correct}")
    
    return {
        'predicted': predicted,
        'correct': correct,
        'is_correct': predicted == correct,
        'full_response': response.content[0].text
    }


if __name__ == "__main__":
    api_key = get_api_key()
    # Load data and get a sample
    mmlu = MMLUData("MMLU")
    data = mmlu.gather_and_format()
    question = mmlu.sample_question()
    result = test_single_question(api_key, question)
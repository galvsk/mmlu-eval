import anthropic
from generate_dataframes import MMLUData
from utils import get_api_key


def test_single_question(api_key, question):
    """Test a single MMLU question using Claude."""
    client = anthropic.Anthropic(api_key=api_key)
    
    prompt = f"""Question: {question['question']}

Options:
A) {question['choices'][0]}
B) {question['choices'][1]}
C) {question['choices'][2]}
D) {question['choices'][3]}

Please respond with just a single letter (A, B, C, or D) representing your answer."""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=5,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Extract predicted answer from content list
    predicted = response.content[0].text.strip()[0]
    correct = chr(65 + int(question['answer']))  # Convert 0-3 to A-D
    
    print(f"\nQuestion: {question['question']}")
    print("\nChoices:")
    for i, choice in enumerate(question['choices']):
        print(f"{chr(65+i)}) {choice}")
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
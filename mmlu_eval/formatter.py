from dataclasses import dataclass
from typing import List, Tuple, Dict
import random


@dataclass
class MMLUPromptDefault:
    question: str
    choices: List[str]
    answer: int
    instructions: str = "Please respond with ONLY the letter (A-D) corresponding to what you believe is the correct answer."
    subject: str = None
   
    def format_question(self) -> Tuple[str, int, Dict[int, int]]:
        letters = ['A', 'B', 'C', 'D']
        choices = "\n".join(
            f"{letter}) {c}" for letter, c in zip(letters, self.choices)
        )
        formatted_question = (
            f"Question: {self.question}\n\n"
            f"Options:\n{choices}\n\n"
            f"{self.instructions}"
        )
        # For default class, each position maps to itself
        position_mapping = {i: i for i in range(len(self.choices))}

        return formatted_question, self.answer, position_mapping
    
@dataclass
class MMLUPromptUpperCase(MMLUPromptDefault):
    def format_question(self) -> Tuple[str, int, Dict[int, int]]:
        formatted_question, answer, position_mapping = super().format_question()
        return formatted_question.upper(), answer, position_mapping 
    
@dataclass
class MMLUPromptRandomCase(MMLUPromptDefault):
    def format_question(self) -> Tuple[str, int, Dict[int, int]]:
        letters = ['A', 'B', 'C', 'D']
        choices = "\n".join(
            f"{letter}) {self.randomize_casing(c)}" for letter, c in zip(letters, self.choices)
        )
        formatted_question = (
            f"{self.randomize_casing('Question')}: {self.randomize_casing(self.question)}\n\n"
            f"{self.randomize_casing('Options')}:\n{choices}\n\n"
            f"{self.instructions}"
        )
        # For default class, each position maps to itself
        position_mapping = {i: i for i in range(len(self.choices))}

        return formatted_question, self.answer, position_mapping
    
    def randomize_casing(self, text: str) -> str:
        return ''.join(
            (c.upper() if random.choice([True, False]) else c.lower())
            for c in text
        )

@dataclass
class MMLUPromptPermuted(MMLUPromptDefault):

    def permute_choices(self) -> Tuple[List[str], int, Dict[int, int]]:
        # position_mapping will map from new position -> original position
        available_positions = list(range(len(self.choices)))
        available_positions.remove(self.answer)  # Remove current answer position
        
        # Pick new position for answer
        new_answer_position = random.choice(available_positions)
        
        # Create shuffled list of remaining original positions
        other_original_positions = [i for i in range(len(self.choices)) if i != self.answer]
        random.shuffle(other_original_positions)
        
        # Build position mapping (new_pos -> original_pos)
        position_mapping = {new_answer_position: self.answer}  # Map new answer position
        
        # Get positions we haven't used for the answer
        remaining_new_positions = [i for i in range(len(self.choices)) if i != new_answer_position]
        
        # Map other positions
        for new_pos, orig_pos in zip(remaining_new_positions, other_original_positions):
            position_mapping[new_pos] = orig_pos
            
        # Create new choices list using the mapping
        new_choices = [self.choices[position_mapping[i]] for i in range(len(self.choices))]
            
        return new_choices, new_answer_position, position_mapping    

    def format_question(self) -> Tuple[str, int, Dict[int, int]]:
        letters = ['A', 'B', 'C', 'D']
        permuted_choices, new_answer, position_mapping = self.permute_choices()
        
        choices = "\n".join(
            f"{letter}) {c}" for letter, c in zip(letters, permuted_choices)
        )
        formatted_question = (
            f"Question: {self.question}\n\n"
            f"Options:\n{choices}\n\n"
            f"{self.instructions}"
        )
        
        return formatted_question, new_answer, position_mapping

@dataclass
class MMLUPromptDuplicateWrong(MMLUPromptDefault):
    instructions: str = "Please respond with ONLY the letter (A-G) corresponding to what you believe is the correct answer."
    
    def format_question(self) -> Tuple[str, int, Dict[int, int]]:
        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        
        # Double all choices an remove one instance of correct answer
        expanded_choices = list(self.choices) * 2
        expanded_choices.pop(self.answer + len(self.choices))
        
        # Shuffle the expanded choices
        random.seed(666)  # For reproducible tests
        random.shuffle(expanded_choices)
        
        # Create position mapping to track where the original correct answer went
        position_mapping = {}
        for new_pos, choice in enumerate(expanded_choices):
            if choice == self.choices[self.answer]:
                # Map the new position back to the original correct answer position
                position_mapping[new_pos] = self.answer
            else:
                # For wrong answers, map to their original positions
                original_pos = list(self.choices).index(choice)
                position_mapping[new_pos] = original_pos
        
        # Format the choices with letters
        choices = "\n".join(
            f"{letter}) {c}" for letter, c in zip(letters, expanded_choices)
        )
        
        formatted_question = (
            f"Question: {self.question}\n\n"
            f"Options:\n{choices}\n\n"
            f"{self.instructions}"
        )
        
        # Find the new position of the correct answer
        new_answer = expanded_choices.index(self.choices[self.answer])
        return formatted_question, new_answer, position_mapping


if __name__ == "__main__":
    test_question = "What is the capital of France?"
    test_choices = ["Paris", "London", "Berlin", "Madrid"]
    correct_answer = 0  # Paris
    
    default_prompt = MMLUPromptDefault(
        question=test_question,
        choices=test_choices,
        answer=correct_answer
    )
    prompt_text, answer_idx, mapping = default_prompt.format_question()
    print("Default Formatter Test:")
    print(prompt_text)
    print(f"Answer index: {answer_idx}")
    print(f"Position mapping: {mapping}")
    assert answer_idx == correct_answer, "Default formatter changed answer position"
    assert mapping[answer_idx] == correct_answer, "Default mapping incorrect"
    print("\nDefault formatter test passed!")
    
    # Test permuted formatter
    permuted_prompt = MMLUPromptPermuted(
        question=test_question,
        choices=test_choices,
        answer=correct_answer
    )
    prompt_text, answer_idx, mapping = permuted_prompt.format_question()
    print("\nPermuted Formatter Test:")
    print(prompt_text)
    print(f"Answer index: {answer_idx}")
    print(f"Position mapping: {mapping}")
    
    # Verify permutation requirements
    assert answer_idx != correct_answer, "Permuted formatter didn't change answer position"
    assert mapping[answer_idx] == correct_answer, "Permuted mapping incorrect"
    assert len(mapping) == len(test_choices), "Mapping missing positions"
    assert set(mapping.values()) == set(range(len(test_choices))), "Mapping values incorrect"
    assert set(mapping.keys()) == set(range(len(test_choices))), "Mapping keys incorrect"
    print("\nPermuted formatter test passed!")

    # Test uppercase formatter
    uppercase_prompt = MMLUPromptUpperCase(
        question=test_question,
        choices=test_choices,
        answer=correct_answer
    )
    prompt_text, answer_idx, mapping = uppercase_prompt.format_question()
    print("\nUppercase Formatter Test:")
    print(prompt_text)
    print(f"Answer index: {answer_idx}")
    print(f"Position mapping: {mapping}")
    
    assert prompt_text.isupper(), "Text is not all uppercase"
    assert answer_idx == correct_answer, "Uppercase formatter changed answer position"
    assert mapping[answer_idx] == correct_answer, "Uppercase mapping incorrect"
    print("\nUppercase formatter test passed!")

    # Test random case formatter
    random_case_prompt = MMLUPromptRandomCase(
        question=test_question,
        choices=test_choices,
        answer=correct_answer
    )
    prompt_text, answer_idx, mapping = random_case_prompt.format_question()
    print("\nRandom Case Formatter Test:")
    print(prompt_text)
    print(f"Answer index: {answer_idx}")
    print(f"Position mapping: {mapping}")

    # Test that case has been modified (at least one char is upper and one is lower)
    text_without_spaces = ''.join(c for c in prompt_text if not c.isspace())
    has_upper = any(c.isupper() for c in text_without_spaces)
    has_lower = any(c.islower() for c in text_without_spaces)
    assert has_upper and has_lower, "Text should have mix of upper and lower case"
    assert answer_idx == correct_answer, "Random case formatter changed answer position"
    assert mapping[answer_idx] == correct_answer, "Random case mapping incorrect"
    print("\nRandom case formatter test passed!")
   
    # Test duplicate wrong answer prompting 
    duplicate_prompt = MMLUPromptDuplicateWrong(
        question=test_question,
        choices=test_choices,
        answer=correct_answer
    )
    
    prompt_text, answer_idx, mapping = duplicate_prompt.format_question()
    print("Duplicate Wrong Answers Formatter Test:")
    print(prompt_text)
    print(f"Answer index: {answer_idx}")
    print(f"Position mapping: {mapping}")
    options = [line.split(') ')[1] for line in prompt_text.split('\n') 
              if line.strip() and line.strip()[0].isalpha() and ') ' in line]
    assert options.count(test_choices[correct_answer]) == 1, "Correct answer should appear exactly once"
    # Test wrong answers appear twice
    for i, choice in enumerate(test_choices):
        if i != correct_answer:
            assert options.count(choice) == 2, f"Wrong answer {choice} should appear exactly twice"
    assert mapping[answer_idx] == correct_answer, "Position mapping doesn't correctly track correct answer"
    print("\nDuplicate wrong answers formatter test passed!")

from dataclasses import dataclass
from typing import List, Tuple, Dict
import random

@dataclass
class MMLUPromptDefault:
    question: str
    choices: List[str]
    answer: int
    instructions: str = "Please respond with ONLY the letter (A-D) corresponding to what you believe is the correct answer."
    
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
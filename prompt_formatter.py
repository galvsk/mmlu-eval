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
class MMLUPromptPermuted(MMLUPromptDefault):
    def permute_choices(self) -> Tuple[List[str], int, Dict[int, int]]:
        position_mapping = {}
        
        available_positions = [i for i in range(len(self.choices)) if i != self.answer]
        new_answer_position = random.choice(available_positions)
        
        other_choices = [i for i in range(len(self.choices)) if i != self.answer]
        other_choices.remove(new_answer_position)
        random.shuffle(other_choices)
        
        position_mapping[new_answer_position] = self.answer
        
        original_positions = [i for i in range(len(self.choices)) if i != self.answer]
        for new_pos, orig_pos in zip(other_choices, original_positions):
            position_mapping[new_pos] = orig_pos
            
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

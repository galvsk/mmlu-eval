import subprocess
from dataclasses import dataclass
from typing import List


def get_api_key():
    cmd = ['security', 'find-generic-password', '-a', 'galvin.s.k@gmail.com', '-s', 'claude-api-key', '-w']
    api_key = subprocess.check_output(cmd).decode('utf-8').strip()
    
    return api_key

@dataclass
class MMLUPromptDefault:
    question: str
    choices: List[str]
    instructions: str = "Please respond with ONLY the letter (A-D) corresponding to what you believe is the correct answer."
    
    def format(self) -> str:
        letters = ['A', 'B', 'C', 'D']
        choices = "\n".join(
            f"{letter}) {c}" for letter, c in zip(letters, self.choices)
        )
        return (
            f"Question: {self.question}\n\n"
            f"Options:\n{choices}\n\n"
            f"{self.instructions}"
        )

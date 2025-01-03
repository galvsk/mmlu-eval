from dataclasses import dataclass
from typing import Dict, Any
from abc import ABC, abstractmethod
import anthropic
from openai import OpenAI
from utils import get_api_key


@dataclass
class ClaudeConfig:
    """Configuration for Claude API."""
    model: str = "claude-3-5-sonnet-20241022"

@dataclass
class DeepseekConfig:
    """Configuration for Deepseek API."""
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    system_prompt: str = "You are a helpful assistant"

class ModelAPI(ABC):
    """Base class for model APIs."""
    @abstractmethod
    def get_completion(self, prompt: str) -> Dict[str, Any]:
        pass

class ClaudeAPI(ModelAPI):
    def __init__(self, config: ClaudeConfig):
        self.config = config
        self.client = anthropic.Anthropic(api_key=get_api_key('claude'))
        
    def get_completion(self, prompt: str) -> Dict[str, Any]:
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=5,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        return {
            'prediction': response.content[0].text[0],
            'logprob': None
        }

class DeepseekAPI(ModelAPI):
    def __init__(self, config: DeepseekConfig):
        self.config = config
        self.client = OpenAI(
            api_key=get_api_key('deepseek'),
            base_url=config.base_url
        )
        
    def get_completion(self, prompt: str) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": prompt}
            ],
            logprobs=True,
            max_tokens=5,
            temperature=0
        )
        return {
            'prediction': response.choices[0].message.content,
            'logprob': response.choices[0].logprobs.content[0].logprob if hasattr(response.choices[0], 'logprobs') else None
        }
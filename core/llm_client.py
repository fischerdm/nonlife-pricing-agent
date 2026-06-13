import json
from anthropic import Anthropic
from typing import Type
from pydantic import BaseModel

SYSTEM_PROMPT = (
    "You are a senior actuary specializing in Non-Life insurance pricing. "
    "You provide actuarially sound, statistically justified analysis. "
    "Your responses are always in valid JSON format as specified in each request, "
    "with no preamble, no markdown fences, and no explanation outside the JSON."
)


class LLMClient:
    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.2,
    ):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def call(self, prompt: str, response_model: Type[BaseModel]) -> BaseModel:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            temperature=self.temperature,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = response.content[0].text
        clean = raw_text.strip().removeprefix("```json").removesuffix("```").strip()
        return response_model.model_validate(json.loads(clean))

import json
from pathlib import Path
from typing import Type

import yaml
from anthropic import Anthropic
from pydantic import BaseModel

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

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
        self._prompt_cache: dict[str, dict] = {}

    def call(self, prompt: str, response_model: Type[BaseModel]) -> BaseModel:
        """Call the LLM with a raw prompt string."""
        return self._call(SYSTEM_PROMPT, prompt, response_model)

    def call_template(
        self,
        agent_name: str,
        section: str,
        response_model: Type[BaseModel],
        **kwargs,
    ) -> BaseModel:
        """Load a prompt template from prompts/<agent_name>.yaml and call the LLM."""
        templates = self._load_prompt_file(agent_name)
        system = templates.get("system", SYSTEM_PROMPT)
        prompt = templates[section].format(**kwargs)
        return self._call(system, prompt, response_model, cache_system=True)

    def _call(
        self,
        system: str,
        prompt: str,
        response_model: Type[BaseModel],
        cache_system: bool = True,
    ) -> BaseModel:
        system_block: dict = {"type": "text", "text": system}
        if cache_system:
            system_block["cache_control"] = {"type": "ephemeral"}

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            temperature=self.temperature,
            system=[system_block],
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = response.content[0].text
        clean = raw_text.strip().removeprefix("```json").removesuffix("```").strip()
        return response_model.model_validate(json.loads(clean))

    def _load_prompt_file(self, agent_name: str) -> dict:
        if agent_name not in self._prompt_cache:
            path = _PROMPTS_DIR / f"{agent_name}.yaml"
            with open(path) as f:
                self._prompt_cache[agent_name] = yaml.safe_load(f)
        return self._prompt_cache[agent_name]

import os

import requests
from dotenv import load_dotenv

from .cache_config import PROJECT_ROOT


class OpenrouterClientError(Exception):
    """Base exception for OpenrouterClient"""


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenrouterClient:
    def __init__(self) -> None:
        # load project environment variables to get api key
        load_dotenv(PROJECT_ROOT / ".env")
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise OpenrouterClientError(
                'OPENROUTER_API_KEY is not set - add it to ".env" at project root'
            )
        self._api_key = api_key

        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self._api_key}"})

    def send_prompt(self, prompt: str, model: str) -> str:
        """Send a single user prompt to the given model and return its text response."""
        response = self.session.post(
            OPENROUTER_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

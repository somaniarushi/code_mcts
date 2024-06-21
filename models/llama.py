import os
import time
from typing import Optional
import requests

MAX_ALLOWED_TRIALS = 7

URL = "https://api.together.xyz/v1/chat/completions"

HEADERS = {
    "accept": "application/json",
    "content-type": "application/json",
    "Authorization": f"Bearer {os.environ.get('TOGETHER_BEARER_TOKEN')}",
}


class LlamaSampler:
    """
    Sample from Together's llama3 chat completion API
    """

    def __init__(
        self,
        model: str = "meta-llama/Llama-3-8b-chat-hf",
        system_message: Optional[str] = None,
        temperature: float = 0.5,
        top_k: int = 40,
        n_generations: int = 1,
        max_tokens: int = 1024,
    ):
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.n_generations = n_generations
        self.image_format = "url"

    def prompt_to_message_format(self, prompt: str):
        return [
        {
            "role": "system",
            "content": "Generate the completion of this code snippet. DO NOT generate anything else except the completion of the code snippet."
        },
        {
            "role": "user",
            "content": prompt
        }]

    def __call__(self, prompt: str) -> str:
        trial = 0
        while trial < MAX_ALLOWED_TRIALS:
            try:
                payload = {
                    "model": self.model,
                    "messages": self.prompt_to_message_format(prompt),
                    # "max_tokens": self.max_tokens,
                    "max_tokens"
                    "temperature": self.temperature,
                    "top_k": self.top_k,
                    "n": self.n_generations,
                }
                response = requests.post(URL, json=payload, headers=HEADERS)
                assert (
                    response.status_code == 200
                ), f"Rate limit detected: {response.text}"
                return response.json()
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
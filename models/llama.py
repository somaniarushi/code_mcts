import os
import time
from typing import Optional, List, Tuple
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
        # model: str = "meta-llama/Llama-3-8b-hf",
        system_message: Optional[str] = None,
        temperature: float = 0.5,
        top_k: int = 40,
        n_generations: int = 1,
        max_tokens: int = 1024,
        logprob: Optional[int] = None,
    ):
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.n_generations = n_generations
        self.image_format = "url"
        self.logprob = logprob

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

    def __call__(self, prompt: str, *, stop_tokens: List[str] = ["```"]) -> str:
        trial = 0
        while trial < MAX_ALLOWED_TRIALS:
            try:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    # "messages": self.prompt_to_message_format(prompt),
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_k": self.top_k,
                    "n": self.n_generations,
                    "repetition_penalty": 1,
                    "stop": stop_tokens,
                }
                if self.logprob is not None:
                    payload["logprobs"] = self.logprob
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

    def get_next_tokens(self, curr_text: str) -> List[Tuple[List[str], List[float]]]:
        """
        Get the next token in the sequence using the model.
        """
        response = self(curr_text, stop_tokens=["\n"])
        logprob_details = [choice["logprobs"] for choice in response["choices"]]
        tokens_and_logprobs = [
            (detail["tokens"], detail["token_logprobs"])
            for detail in logprob_details
        ]
        return tokens_and_logprobs

    def generate(self, prompt: str) -> str:
        """
        Generate the completion of the code snippet
        and return the first full generated text.
        """
        output = self(prompt)
        return output["choices"][0]["message"]["content"]
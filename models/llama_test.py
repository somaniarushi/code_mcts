from models.llama import LlamaSampler

class TestLlama:
    def test_llama_gen_basic(self) -> None:
        sampler = LlamaSampler()
        generation = sampler("Hello!")["choices"][0]["message"]["content"]
        assert generation == (
            "Hello! It's nice to meet you. "
            "Is there something I can help you with, "
            "or would you like to chat?"
        )
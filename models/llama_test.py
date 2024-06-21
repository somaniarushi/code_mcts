from models.llama import LlamaSampler

class TestLlama:
    def test_llama_gen_basic(self) -> None:
        sampler = LlamaSampler()
        generation = sampler("Hello!")["choices"][0]["message"]["content"]
        assert generation is not None

    def test_llama_logprobs(self) -> None:
        sampler = LlamaSampler(logprob=1, n_generations=3, max_tokens=1)
        tokens_and_logprobs = sampler.get_next_token("Hello!")
        assert len(tokens_and_logprobs) == 3
        assert all(token == "I" for token, logprob in tokens_and_logprobs)
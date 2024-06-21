from tqdm import tqdm
from human_eval.data import write_jsonl, read_problems
from models.llama import LlamaSampler

def run_eval(outpath: str, n_samples: int) -> None:
    problems = read_problems()
    sampler = LlamaSampler(n_generations=n_samples)

    samples = []
    for task_id, problem in tqdm(problems.items()):
        responses = sampler(problem["prompt"])
        for response in responses["choices"]:
            samples.append(dict(task_id=task_id, completion=response["message"]["content"]))

    write_jsonl(f"{outpath}/samples_llama_{n_samples}.jsonl", samples)
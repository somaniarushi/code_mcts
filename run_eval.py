from __future__ import annotations

from human_eval.data import read_problems
from human_eval.data import write_jsonl
from tqdm import tqdm

from models.llama import LlamaSampler
from run_mcts import append_to_jsonl
from run_mcts import evaluate_completion


def run_eval(outpath: str, n_samples: int) -> None:
    problems = read_problems()
    sampler = LlamaSampler(n_generations=1)

    # If the file doesn't exist, create it
    filepath = f'{outpath}/samples_llama_{n_samples}.jsonl'
    with open(filepath, 'w') as f:
        f.write('')

    samples = []
    for task_id, problem in tqdm(problems.items()):
        for i in range(n_samples):
            print(f'Attempt {i + 1} for task {task_id}')
            response = sampler(problem['prompt'])

            completion = response['choices'][0]['message']['content']
            assert isinstance(
                completion, str,
            ), f'Expected completion to be a string, but got {type(completion)} {completion=}'

            reward = evaluate_completion(completion, problem)

            append_to_jsonl(
                output_jsonl_path=filepath, task_id=task_id,
                completion=completion, reward=reward, rollout_counts_for_solution=-1,
            )
            if reward == 1:
                print(f'Found a valid completion for task {task_id}')
                break
            samples.append(dict(task_id=task_id, completion=completion))

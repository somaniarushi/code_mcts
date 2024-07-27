from __future__ import annotations

import json
import math
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from human_eval.data import read_problems
from tqdm import tqdm

from data_processing.clean_jsonl import clean_completion
from evaluation import stats_execute
from mcts.tree import Node
from models.llama import LlamaSampler


def print_tree(root: Node, problem_str: str, depth: int = 0) -> None:
    tab = '\t' * depth
    print(f"{tab}{root.state.replace(problem_str, '').strip()}\n-----\n")
    for child in root.children:
        print_tree(child, problem_str, depth + 1)


def evaluate_completion(completion: str, problem: dict[str, Any]) -> float:
    # response = check_correctness_fine_grained(problem, completion, timeout=3.0)
    response = stats_execute(task_id=problem['task_id'], completion=completion)
    return response['pass_rate']


def early_exit_reward_or_none(curr_text: str, solution_to_reward: dict[str, int]) -> int | None:
    for solution in solution_to_reward:
        if solution.startswith(curr_text):
            return solution_to_reward[solution]
    return None


def append_to_jsonl(output_jsonl_path: str, task_id: str, completion: str, reward: int, rollout_counts_for_solution: int) -> None:
    with open(output_jsonl_path, 'a') as f:
        f.write(
            json.dumps(
                {
                    'task_id': task_id,
                    'completion': completion,
                    'reward': reward,
                    'rollout_count': rollout_counts_for_solution,
                },
            ) + '\n',
        )


def main(output_jsonl_path: str, max_rollouts: int) -> None:
    next_token_generator = LlamaSampler(
        max_tokens=100,
        n_generations=10,
        logprob=1,
        temperature=1,
    )
    rollout_generator = LlamaSampler(max_tokens=200)
    problems = read_problems()

    for task_id, problem in tqdm(problems.items()):

        solution_to_reward = {}

        prompt = problem['prompt']
        root = Node('', math.log(1), prompt, None)  # <PD> is the root node
        rollout_counts_for_solution = -1
        for i in range(max_rollouts):
            print(f'Starting rollout {i + 1} for task {task_id}')
            curr = root  # Start at the root

            # Add a visit to the current node
            curr.visits += 1

            # Part 1: Selection
            while len(curr.children) > 0:
                # Select the best child node
                curr, _ = curr.get_child_by_ucb_confidence()
                # Add a visit to the selected node
                curr.visits += 1

            # Part 2: Expansion
            next_tokens_and_logprobs: list[tuple[list[str], list[float]]] = (
                next_token_generator.get_next_tokens(curr.state)
            )
            next_sentences = [
                ''.join(tokens)
                for tokens, _ in next_tokens_and_logprobs
            ]
            summed_logprobs = []
            for _, logprob in next_tokens_and_logprobs:
                summed_logprobs.append(sum(logprob))
            next_tokens_and_logprobs = list(
                zip(next_sentences, summed_logprobs),
            )
            # Optimization, dedup on the next tokens
            deduped_next_tokens_and_logprobs = []
            seen = set()
            for token, logprob in next_tokens_and_logprobs:
                if token not in seen:
                    deduped_next_tokens_and_logprobs.append((token, logprob))
                    seen.add(token)
            next_tokens_and_logprobs = deduped_next_tokens_and_logprobs

            # print(f"Next tokens: {next_tokens} with curr_state: \n```\n{curr.state}\n```\n")
            # Create child nodes
            child_nodes = [
                Node(token, logprob, curr.state + token, curr)
                for token, logprob in next_tokens_and_logprobs
            ]
            curr.children = child_nodes

            # Part 3: Evaluation
            # Early exit if a solution is found
            reward = early_exit_reward_or_none(curr.state, solution_to_reward)
            if reward is None:
                rollout = curr.state + rollout_generator.generate(curr.state)
                # Remove the prompt for evaluation
                completion = rollout.replace(prompt, '')
                completion = clean_completion(completion)
                # 1 if correct, 0 otherwise
                reward = evaluate_completion(completion, problem)
                if reward == 1:
                    print(f'Found solution in {i + 1} steps!')
                    solution_to_reward[rollout] = reward
                else:
                    pass
                    # print(f"===============================")
                    # print(f"Prefix for task {task_id}:\n```\n{curr.state}\n```\n")
                    # print(f"----")
                    # print(f"Completion for task {task_id}:\n```\n{completion}\n```\n UNSUCCESSFUL with reward {reward}")
                    # print(f"===============================")
                # TODO: Can be more fine-grained?

            # Part 4: Backpropagation
            curr.backprop(reward)

            if reward == 1:  # Found the correct completion
                rollout_counts_for_solution = i + 1
                break

        # Get the best completion
        if solution_to_reward:
            best_completion = max(
                solution_to_reward,
                key=solution_to_reward.get,
            )
            best_reward = solution_to_reward[best_completion]
        else:
            best_completion = ''
            best_reward = 0

        append_to_jsonl(
            output_jsonl_path, task_id, best_completion,
            best_reward, rollout_counts_for_solution,
        )
        print_tree(root, problem['prompt'])

    print(f'Output written to {output_jsonl_path}')
    print(f'Done!')

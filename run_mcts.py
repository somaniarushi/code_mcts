from __future__ import annotations

import json
import math
import re
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


def get_reward_from_llm_value_network(state: str, model: LlamaSampler) -> float:
    """
    Asks the model to return a value between 0 and 100 for a given state. Then extracts that value from the response and
    returns it as the reward. If no value is extracted, returns 0.
    """
    SCORE_MATCH_REGEX = r'Score:\s*(\d+)'
    prompt = state + '\nGiven this is how far a student has gotten in their solution to the problem, score them on a scale of 0 to 100. Explain your reasoning then leave your score as Score: <score>. Be concise and clear.'
    response = model(prompt)
    completion = response['choices'][0]['message']['content']
    score_match = re.search(SCORE_MATCH_REGEX, completion)
    if score_match:
        integer_score = int(score_match.group(1))
        score = integer_score / 100
    else:
        score = 0.0
    print(f'Score: {score}')
    return score


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

    # Get the last 30 tasks
    problems = dict(list(problems.items())[-20:])  # To test fast. Revert.

    for task_id, problem in tqdm(problems.items()):

        solution_to_reward = {}

        prompt = 'Solve the problem below by completing the given function. Once completed, end your generation with ```. Do not generate more problems or tests.\n```python\n\n' + \
            problem['prompt']
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
            next_tokens_and_logprobs: list[
                tuple[list[str], list[float]]
            ] = next_token_generator.get_next_tokens(curr.state)
            # If any are None, skip this rollout
            if any(x is None or y is None for x, y in next_tokens_and_logprobs):
                print(f'Skipping rollout {i + 1} for task {task_id}')
                continue

            next_sentences = []
            summed_logprobs = []
            try:
                next_sentences = [
                    ''.join(tokens)
                    for tokens, _ in next_tokens_and_logprobs
                ]
                summed_logprobs = [
                    sum(logprob)
                    for _, logprob in next_tokens_and_logprobs
                ]
                next_tokens_and_logprobs = list(
                    zip(next_sentences, summed_logprobs),
                )
            except Exception as e:
                print(f'Error in getting next tokens: {e}')
                continue

            # Optimization, dedup on the next tokens
            deduped_next_tokens_and_logprobs = []
            seen = set()
            for token, logprob in next_tokens_and_logprobs:
                if token not in seen:
                    deduped_next_tokens_and_logprobs.append((token, logprob))
                    seen.add(token)
            next_tokens_and_logprobs = deduped_next_tokens_and_logprobs

            # Create child nodes
            child_nodes = [
                Node(token, logprob, curr.state + token, curr) for token, logprob in next_tokens_and_logprobs
            ]
            curr.children = child_nodes

            # Part 3: Evaluation
            # Early exit if a solution is found
            reward = early_exit_reward_or_none(curr.state, solution_to_reward)
            if reward is None:
                direct_reward = evaluate_completion(
                    clean_completion(curr.state.replace(prompt, '')), problem,
                )
                if direct_reward == 1:
                    print(f'Found solution in {i + 1} steps!')
                    solution_to_reward[curr.state] = direct_reward
                    reward = direct_reward
                else:
                    # Predict the reward using LLM as a value network
                    reward = get_reward_from_llm_value_network(
                        curr.state, rollout_generator,
                    )
                    print(f'===============================')
                    print(
                        f'Prefix for task {task_id}:\n```\n{curr.state}\n```\n',
                    )
                    print(f'----')
                    print(
                        f'State for task {task_id}:\n```\n{curr.state}\n```\n UNSUCCESSFUL with model reward {reward} | direct reward {direct_reward}',
                    )
                    print(f'===============================')

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

    print(f'Output written to {output_jsonl_path}')
    print(f'Done!')

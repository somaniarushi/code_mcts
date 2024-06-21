from tqdm import tqdm
import math
import json
from typing import List, Tuple, Dict, Any, Optional
from human_eval.data import read_problems
from evaluation import stats_execute

from models.llama import LlamaSampler
from mcts.tree import Node
from data_processing.clean_jsonl import clean_completion


def evaluate_completion(completion: str, problem: Dict[str, Any]) -> float:
    # response = check_correctness_fine_grained(problem, completion, timeout=3.0)
    response = stats_execute(task_id=problem["task_id"], completion=completion)
    return response["pass_rate"]

def early_exit_reward_or_none(curr_text: str, solution_to_reward: Dict[str, int]) -> Optional[int]:
    for solution in solution_to_reward:
        if solution.startswith(curr_text):
            return solution_to_reward[solution]
    return None

def append_to_jsonl(output_jsonl_path: str, task_id: str, completion: str, reward: int, rollout_counts_for_solution: int) -> None:
    with open(output_jsonl_path, "a") as f:
        f.write(
            json.dumps(
                {
                    "task_id": task_id,
                    "completion": completion,
                    "reward": reward,
                    "rollout_count": rollout_counts_for_solution
                }
            ) + "\n")

def main(output_jsonl_path: str, max_rollouts: int) -> None:
    next_token_generator = LlamaSampler(max_tokens=1, n_generations=3, logprob=1, temperature=1)
    rollout_generator = LlamaSampler(max_tokens=200)
    problems = read_problems()

    for task_id, problem in tqdm(problems.items()):

        solution_to_reward = {}

        prompt = problem["prompt"]
        root = Node("", math.log(1), prompt, None) # <PD> is the root node
        rollout_counts_for_solution = -1
        for i in range(max_rollouts):
            print(f"Starting rollout {i + 1} for task {task_id}")
            curr = root # Start at the root

            # Add a visit to the current node
            curr.visits += 1

            #### Part 1: Selection
            while len(curr.children) > 0:
                # Select the best child node
                curr, _ = curr.get_child_by_ucb_confidence()
                # Add a visit to the selected node
                curr.visits += 1

            ##### Part 2: Expansion
            next_tokens_and_logprobs: List[Tuple[str, float]] = (
                next_token_generator.get_next_token(curr.state)
            )
            # Optimization: Dedup the next_tokens
            next_tokens_and_logprobs = list(set(next_tokens_and_logprobs))
            next_tokens = list(map(lambda x: x[0], next_tokens_and_logprobs))
            # print(f"Next tokens: {next_tokens} with curr_state: \n```\n{curr.state}\n```\n")
            # Create child nodes
            child_nodes = [
                Node(token, logprob, curr.state + token, curr)
                for token, logprob in next_tokens_and_logprobs
            ]
            curr.children = child_nodes

            ##### Part 3: Evaluation
            # Early exit if a solution is found
            reward = early_exit_reward_or_none(curr.state, solution_to_reward)
            if reward is None:
                rollout = curr.state + rollout_generator.generate(curr.state)
                completion = rollout.replace(prompt, "") # Remove the prompt for evaluation
                completion = clean_completion(completion)
                reward = evaluate_completion(completion, problem) # 1 if correct, 0 otherwise
                if reward == 1:
                    print(f"Found solution in {i + 1} steps!")
                    solution_to_reward[rollout] = reward
                else:
                    print(f"Completion for task {task_id}:\n```\n{completion}\n```\n UNSUCCESSFUL with reward {reward}")
                # TODO: Can be more fine-grained?

            ##### Part 4: Backpropagation
            curr.backprop(reward)

            if reward == 1: # Found the correct completion
                rollout_counts_for_solution = i + 1
                break

        # Get the best completion
        if solution_to_reward:
            best_completion = max(solution_to_reward, key=solution_to_reward.get)
            best_reward = solution_to_reward[best_completion]
        else:
            best_completion = ""
            best_reward = 0

        append_to_jsonl(output_jsonl_path, task_id, best_completion, best_reward, rollout_counts_for_solution)
    print(f"Output written to {output_jsonl_path}")
    print(f"Done!")

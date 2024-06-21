import itertools
import math
from typing import List, Tuple

C_BASE = 10
C = 4

# Source: https://github.com/rmshin/llm-mcts
class Node:
    id_iter = itertools.count()

    def __init__(self, label: str, logprob: float, state: str, parent: "Node") -> None:
        """
        Initializes a new node in the search tree.

        Args:
            label: The token label text.
            logprob: The log probability of the token.
            state: The full generated text.
            parent: The parent node.
        """
        self.value = 0 # Will be updated through backprop
        self.prob = math.exp(logprob)
        self.state = state

        self.children = []
        self.parent = parent

        self.visits = 0

    def backprop(self, value: float) -> None:
        """
        Propagates the value back up the tree.

        Args:
            value: The value to propagate.
        """
        if value > self.value:
            self.value = value
            if self.parent is not None:
                self.parent.backprop(value)

    def __repr__(self) -> str:
        return f"Node({self.state}) | {self.prob:.2f}"

    def get_child_by_ucb_confidence(self) -> Tuple["Node", float]:
        parent, children = self, self.children
        parent_visits = parent.visits
        # ß(s) = log((s.visits + c_base + 1) / c_base) + c
        beta = math.log((parent_visits + C_BASE + 1) / C_BASE) + C

        max_ucb = float('-infinity')
        max_node = None
        for child in children:
            ucb = (
                child.value +
                beta * child.prob * (
                    math.sqrt(math.log(parent_visits)) / (1 + child.visits)
                )
            )
            if ucb > max_ucb:
                max_ucb = ucb
                max_node = child
        return max_node, max_ucb
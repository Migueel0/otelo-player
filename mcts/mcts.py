import math
import random

import numpy as np

from otelo.evaluation import evaluate_model, heuristic_value, winner_to_value_for_player

class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_moves = state.get_valid_moves(state.current_player)
        if not self.untried_moves and not state.is_game_over():
            self.untried_moves = [None]

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def is_terminal(self):
        return self.state.is_game_over()

def get_best_child(node, exploration_weight=1.414):
    best_value = -float('inf')
    best_nodes = []

    for child in node.children:
        if child.visits == 0:
            ucb1 = float('inf')
        else:
            avg_value = child.value / child.visits
            signed_value = -avg_value
            ucb1 = signed_value + exploration_weight * math.sqrt(math.log(max(1, node.visits)) / child.visits)

        if ucb1 > best_value:
            best_value = ucb1
            best_nodes = [child]
        elif ucb1 == best_value:
            best_nodes.append(child)

    return random.choice(best_nodes)

def _evaluate_leaf(state, neural_network):
    if state.is_game_over():
        return winner_to_value_for_player(state.current_player, state.get_winner())
    if neural_network is not None:
        return evaluate_model(neural_network, state)
    return heuristic_value(state)


def mcts(root_state, num_iterations, neural_network=None, node_counter=None):
    root = MCTSNode(root_state)
    if node_counter is not None:
        node_counter[0] += 1

    for _ in range(num_iterations):
        node = root

        # 1. Selection
        while node.is_fully_expanded() and not node.is_terminal():
            node = get_best_child(node)

        # 2. Expansion
        if not node.is_terminal():
            move = node.untried_moves.pop()
            next_state = node.state.copy()
            if move is not None:
                next_state.make_move(move, next_state.current_player)
            else:
                next_state.current_player = 3 - next_state.current_player

            child_node = MCTSNode(next_state, parent=node, move=move)
            node.children.append(child_node)
            node = child_node

            if node_counter is not None:
                node_counter[0] += 1

        base_reward = _evaluate_leaf(node.state, neural_network)

        # 4. Backpropagation
        curr = node
        sign = 1.0
        while curr is not None:
            curr.visits += 1
            curr.value += sign * base_reward
            sign *= -1.0
            curr = curr.parent

    if not root.children:
        return None

    best_child = max(root.children, key=lambda c: c.visits)
    return best_child.move


def choose_move(state, num_iterations, neural_network=None, node_counter=None):
    return mcts(state, num_iterations, neural_network=neural_network, node_counter=node_counter)

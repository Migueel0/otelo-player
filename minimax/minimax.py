import math

import numpy as np

from otelo.evaluation import canonical_board, heuristic_value, winner_to_value_for_player


def utility(state, model):
    if state.is_game_over():
        return winner_to_value_for_player(state.current_player, state.get_winner())

    if model is not None:
        board_array = canonical_board(state.board, state.current_player).reshape(1, 8, 8, 3)
        return float(model.predict(board_array, verbose=0).flatten()[0])

    return heuristic_value(state)


def _order_moves(state, moves, model=None):
    scored_moves = []
    for move in moves:
        child = state.copy()
        child.make_move(move, state.current_player)
        scored_moves.append((utility(child, model), move))
    scored_moves.sort(key=lambda item: item[0], reverse=True)
    return [move for _, move in scored_moves]


def alphabeta(state, depth, alpha, beta, player, maximizing_player, neural_network, node_counter=None):
    if node_counter is not None:
        node_counter[0] += 1

    if depth == 0 or state.is_game_over():
        return None, utility(state, neural_network)

    valid_moves = state.get_valid_moves(player)
    if not valid_moves:
        return alphabeta(state, depth, alpha, beta, 3 - player, not maximizing_player, neural_network, node_counter)

    if maximizing_player:
        max_eval = -math.inf
        best_move = None
        ordered_moves = _order_moves(state, valid_moves, neural_network)
        for move in ordered_moves:
            state_copy = state.copy()
            state_copy.make_move(move, player)
            _, eval_score = alphabeta(state_copy, depth - 1, alpha, beta, 3 - player, False, neural_network, node_counter)
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return best_move, max_eval

    min_eval = math.inf
    best_move = None
    ordered_moves = _order_moves(state, valid_moves, neural_network)
    for move in ordered_moves:
        state_copy = state.copy()
        state_copy.make_move(move, player)
        _, eval_score = alphabeta(state_copy, depth - 1, alpha, beta, 3 - player, True, neural_network, node_counter)
        if eval_score < min_eval:
            min_eval = eval_score
            best_move = move
        beta = min(beta, eval_score)
        if beta <= alpha:
            break
    return best_move, min_eval


def choose_move(state, depth, neural_network=None, node_counter=None):
    maximizing_player = state.current_player == 1
    move, _ = alphabeta(state, depth, -math.inf, math.inf, state.current_player, maximizing_player, neural_network, node_counter)
    return move
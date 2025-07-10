import numpy as np


def alphabeta(state, depth, alpha, beta, player, maximizing_player, neural_network, node_counter=None):
    if node_counter is not None:
        node_counter[0] += 1

    if depth == 0 or state.is_game_over():
        return None, utility(state, neural_network)

    valid_moves = state.get_valid_moves(player)
    if not valid_moves:
        return alphabeta(state, depth, alpha, beta, 3 - player, not maximizing_player, neural_network, node_counter)

    if maximizing_player:
        max_eval = float('-inf')
        best_move = None
        for move in valid_moves:
            state_copy = state.copy()
            state_copy.make_move(move, player)
            _, eval = alphabeta(state_copy, depth - 1, alpha, beta, 3 - player, False, neural_network, node_counter)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return best_move, max_eval
    else:
        min_eval = float('inf')
        best_move = None
        for move in valid_moves:
            state_copy = state.copy()
            state_copy.make_move(move, player)
            _, eval = alphabeta(state_copy, depth - 1, alpha, beta, 3 - player, True, neural_network, node_counter)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return best_move, min_eval

def utility(state, model):
    board_array = np.array(state.board, dtype=np.float32).reshape(1, 8, 8, 1)
    return model.predict(board_array).flatten()[0]
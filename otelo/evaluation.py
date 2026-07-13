import numpy as np


CORNER_COORDS = ((0, 0), (0, 7), (7, 0), (7, 7))


def canonical_board(board, current_player):
    board_array = np.asarray(board, dtype=np.float32)
    if current_player == 1:
        black_plane = (board_array == 1).astype(np.float32)
        white_plane = (board_array == 2).astype(np.float32)
    else:
        black_plane = (board_array == 2).astype(np.float32)
        white_plane = (board_array == 1).astype(np.float32)

    current_plane = np.ones((8, 8), dtype=np.float32)
    return np.stack((black_plane, white_plane, current_plane), axis=-1)


def encode_board(board, current_player):
    return canonical_board(board, current_player)


def encode_state(state):
    return encode_board(state.board, state.current_player)


def winner_to_value_for_player(player, winner):
    if winner == 0:
        return 0.0
    return 1.0 if winner == player else -1.0


def winner_to_value(winner):
    return winner_to_value_for_player(1, winner)


def evaluate_model(model, state):
    encoded = encode_state(state)[np.newaxis, ...]
    prediction = model(encoded, training=False).numpy().reshape(-1)[0]
    return float(prediction)


def heuristic_value(state):
    board = canonical_board(state.board, state.current_player)

    black_count = np.count_nonzero(board[..., 0] == 1)
    white_count = np.count_nonzero(board[..., 1] == 1)
    disc_total = max(1, black_count + white_count)
    disc_diff = (black_count - white_count) / disc_total

    black_corners = sum(1 for row, col in CORNER_COORDS if board[row, col, 0] == 1)
    white_corners = sum(1 for row, col in CORNER_COORDS if board[row, col, 1] == 1)
    corner_diff = (black_corners - white_corners) / 4.0

    black_mobility = len(state.get_valid_moves(1))
    white_mobility = len(state.get_valid_moves(2))
    mobility_total = max(1, black_mobility + white_mobility)
    mobility_diff = (black_mobility - white_mobility) / mobility_total

    edge_mask = np.zeros((8, 8), dtype=bool)
    edge_mask[0, :] = True
    edge_mask[7, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, 7] = True
    for row, col in CORNER_COORDS:
        edge_mask[row, col] = False

    black_edges = np.count_nonzero((board[..., 0] > 0) & edge_mask)
    white_edges = np.count_nonzero((board[..., 1] > 0) & edge_mask)
    edge_total = max(1, black_edges + white_edges)
    edge_diff = (black_edges - white_edges) / edge_total

    score = (
        3.5 * corner_diff
        + 1.8 * mobility_diff
        + 0.8 * edge_diff
        + 0.7 * disc_diff
    )
    return float(np.tanh(1.5 * score))
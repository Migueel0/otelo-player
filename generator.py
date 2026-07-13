import argparse
import os
import random
import sys

import numpy as np

from mcts.mcts import choose_move as choose_mcts_move
from minimax.minimax import choose_move as choose_minimax_move
from otelo.evaluation import encode_state, winner_to_value_for_player
from otelo.otelo import OthelloGame


def print_progress(current, total, bar_length=40):
    percent = float(current) / total
    arrow = '-' * max(0, int(round(percent * bar_length) - 1)) + '>'
    spaces = ' ' * max(0, bar_length - len(arrow))
    sys.stdout.write(f'\rProgreso: [{arrow}{spaces}] {int(percent * 100)}%')
    sys.stdout.flush()


def get_symmetries(board):
    syms = []
    for i in range(4):
        rotated = np.rot90(board, i, axes=(0, 1))
        syms.append(rotated)
        syms.append(np.fliplr(rotated))
    return syms


def parse_args():
    parser = argparse.ArgumentParser(description='Genera datos de autojuego para Othello.')
    parser.add_argument('--agent', choices=['mcts', 'minimax', 'random'], default='mcts')
    parser.add_argument('--games', type=int, default=1000)
    parser.add_argument('--output', default='data/othello_data.npz')
    parser.add_argument('--mcts-iterations', type=int, default=60)
    parser.add_argument('--minimax-depth', type=int, default=3)
    return parser.parse_args()


def choose_agent_move(game, agent, mcts_iterations, minimax_depth):
    if agent == 'mcts':
        return choose_mcts_move(game, mcts_iterations, neural_network=None)
    if agent == 'minimax':
        return choose_minimax_move(game, minimax_depth, neural_network=None)

    valid_moves = game.get_valid_moves(game.current_player)
    if not valid_moves:
        return None
    return random.choice(valid_moves)


def generate_game_data(agent='mcts', num_games=1000, output_path='data/othello_data.npz', mcts_iterations=60, minimax_depth=3):
    boards = []
    labels = []

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    for game_index in range(num_games):
        game = OthelloGame()
        states = []
        players = []

        while not game.is_game_over():
            states.append(encode_state(game))
            players.append(game.current_player)
            valid_moves = game.get_valid_moves(game.current_player)
            if not valid_moves:
                game.current_player = 3 - game.current_player
                continue

            move = choose_agent_move(game, agent, mcts_iterations, minimax_depth)
            if move is None:
                move = random.choice(valid_moves)
            game.make_move(move, game.current_player)

        winner = game.get_winner()
        for state, player in zip(states, players):
            label = winner_to_value_for_player(player, winner)
            for symmetry in get_symmetries(state):
                boards.append(symmetry.astype(np.float32))
                labels.append(label)

        print_progress(game_index + 1, num_games)

    print()
    boards = np.asarray(boards, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    np.savez_compressed(output_path, boards=boards, labels=labels)
    print(f'Guardados {len(boards)} estados en {output_path}')


if __name__ == '__main__':
    args = parse_args()
    generate_game_data(
        agent=args.agent,
        num_games=args.games,
        output_path=args.output,
        mcts_iterations=args.mcts_iterations,
        minimax_depth=args.minimax_depth,
    )
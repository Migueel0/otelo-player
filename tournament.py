import argparse
from collections import Counter
import random

from tensorflow import keras

from mcts.mcts import choose_move as choose_mcts_move
from minimax.minimax import choose_move as choose_minimax_move
from otelo.otelo import OthelloGame


def parse_args():
    parser = argparse.ArgumentParser(description='Compara una red con MCTS contra una red con Minimax.')
    parser.add_argument('--mcts-model', default='othello_model.keras')
    parser.add_argument('--minimax-model', default='minimax_model.keras')
    parser.add_argument('--games', type=int, default=20)
    parser.add_argument('--mcts-iterations', type=int, default=300)
    parser.add_argument('--minimax-depth', type=int, default=3)
    return parser.parse_args()


def play_game(mcts_model, minimax_model, mcts_black=True, mcts_iterations=300, minimax_depth=3):
    game = OthelloGame()

    while not game.is_game_over():
        if game.current_player == 1:
            move = choose_mcts_move(game, mcts_iterations, neural_network=mcts_model) if mcts_black else choose_minimax_move(game, minimax_depth, neural_network=minimax_model)
        else:
            move = choose_minimax_move(game, minimax_depth, neural_network=minimax_model) if mcts_black else choose_mcts_move(game, mcts_iterations, neural_network=mcts_model)

        valid_moves = game.get_valid_moves(game.current_player)
        if not valid_moves:
            game.current_player = 3 - game.current_player
            continue

        if move is None:
            move = valid_moves[0]

        game.make_move(move, game.current_player)

    return game.get_winner()


def build_starting_colors(num_games):
    mcts_black_games = num_games // 2
    mcts_white_games = num_games // 2

    starting_colors = [True] * mcts_black_games + [False] * mcts_white_games
    if num_games % 2 == 1:
        starting_colors.append(random.choice([True, False]))

    random.shuffle(starting_colors)
    return starting_colors


def main():
    args = parse_args()
    mcts_model = keras.models.load_model(args.mcts_model, compile=False)
    minimax_model = keras.models.load_model(args.minimax_model, compile=False)

    results = Counter()
    mcts_results = Counter()
    minimax_results = Counter()
    starting_colors = build_starting_colors(args.games)

    for mcts_black in starting_colors:
        winner = play_game(
            mcts_model,
            minimax_model,
            mcts_black=mcts_black,
            mcts_iterations=args.mcts_iterations,
            minimax_depth=args.minimax_depth,
        )

        if winner == 0:
            results['draws'] += 1
            mcts_results['draws'] += 1
            minimax_results['draws'] += 1
        elif (winner == 1 and mcts_black) or (winner == 2 and not mcts_black):
            results['mcts_wins'] += 1
            mcts_results['wins'] += 1
            minimax_results['losses'] += 1
        else:
            results['minimax_wins'] += 1
            minimax_results['wins'] += 1
            mcts_results['losses'] += 1

        if mcts_black:
            mcts_results['black_games'] += 1
            minimax_results['white_games'] += 1
            if winner == 1:
                mcts_results['black_wins'] += 1
                minimax_results['white_losses'] += 1
            elif winner == 2:
                mcts_results['black_losses'] += 1
                minimax_results['white_wins'] += 1
            else:
                mcts_results['black_draws'] += 1
                minimax_results['white_draws'] += 1
        else:
            mcts_results['white_games'] += 1
            minimax_results['black_games'] += 1
            if winner == 2:
                mcts_results['white_wins'] += 1
                minimax_results['black_losses'] += 1
            elif winner == 1:
                mcts_results['white_losses'] += 1
                minimax_results['black_wins'] += 1
            else:
                mcts_results['white_draws'] += 1
                minimax_results['black_draws'] += 1

    print('Resultados del torneo:')
    print(f"MCTS total: {results['mcts_wins']}")
    print(f"Minimax total: {results['minimax_wins']}")
    print(f"Empates totales: {results['draws']}")

    total = max(1, args.games)
    print(f"Tasa de victoria total MCTS: {results['mcts_wins'] / total:.3f}")
    print(f"Tasa de victoria total Minimax: {results['minimax_wins'] / total:.3f}")

    mcts_black_games = max(1, mcts_results['black_games'])
    mcts_white_games = max(1, mcts_results['white_games'])
    minimax_black_games = max(1, minimax_results['black_games'])
    minimax_white_games = max(1, minimax_results['white_games'])

    print('\nDesglose por color:')
    print(f"MCTS jugando como negro: {mcts_results['black_wins']} victorias, {mcts_results['black_losses']} derrotas, {mcts_results['black_draws']} empates")
    print(f"MCTS jugando como blanco: {mcts_results['white_wins']} victorias, {mcts_results['white_losses']} derrotas, {mcts_results['white_draws']} empates")
    print(f"Minimax jugando como negro: {minimax_results['black_wins']} victorias, {minimax_results['black_losses']} derrotas, {minimax_results['black_draws']} empates")
    print(f"Minimax jugando como blanco: {minimax_results['white_wins']} victorias, {minimax_results['white_losses']} derrotas, {minimax_results['white_draws']} empates")

    print('\nPorcentajes por color:')
    print(f"MCTS como negro: {mcts_results['black_wins'] / mcts_black_games:.3f}")
    print(f"MCTS como blanco: {mcts_results['white_wins'] / mcts_white_games:.3f}")
    print(f"Minimax como negro: {minimax_results['black_wins'] / minimax_black_games:.3f}")
    print(f"Minimax como blanco: {minimax_results['white_wins'] / minimax_white_games:.3f}")


if __name__ == '__main__':
    main()
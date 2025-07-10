import numpy as np
from otelo.otelo import OthelloGame
import sys

def print_progress(current, total, bar_length=40):
    percent = float(current) / total
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write(f'\rProgreso: [{arrow}{spaces}] {int(percent*100)}%')
    sys.stdout.flush()

def generate_game_data(num_games=100):
    boards = []
    labels = []
    for i in range(num_games):
        game = OthelloGame()
        states = []
        players = []
        while not game.is_game_over():
            board_copy = np.array(game.board)
            states.append(board_copy)
            players.append(game.current_player)
            valid_moves = game.get_valid_moves(game.current_player)
            if not valid_moves:
                game.current_player = 3 - game.current_player
                continue
            move = valid_moves[np.random.randint(len(valid_moves))]
            game.make_move(move, game.current_player)
        winner = game.get_winner()
        for state, player in zip(states, players):
            if winner == 0:
                label = 0
            elif winner == player:
                label = 1
            else:
                label = -1
            boards.append(state)
            labels.append(label)
        print_progress(i + 1, num_games)
    print()
    boards = np.array(boards)
    labels = np.array(labels)
    np.savez('data/othello_data.npz', boards=boards, labels=labels)
    print(f"Guardados {len(boards)} estados en data/othello_data.npz")

if __name__ == "__main__":
    generate_game_data(num_games=100)
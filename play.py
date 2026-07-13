import argparse
import random
import time

from tensorflow import keras

from mcts.mcts import choose_move as choose_mcts_move
from minimax.minimax import choose_move as choose_minimax_move
from otelo.otelo import OthelloGame

def print_board(board):
    print("  0 1 2 3 4 5 6 7")
    for i in range(8):
        print(i, end=" ")
        for j in range(8):
            if board[i][j] == 0:
                print(".", end=" ")
            elif board[i][j] == 1:
                print("N", end=" ")
            else:
                print("B", end=" ")
        print()


def parse_args():
    parser = argparse.ArgumentParser(description='Juega una partida de Othello contra un agente entrenado.')
    parser.add_argument('--agent', choices=['mcts', 'minimax'], default='mcts')
    parser.add_argument('--model', default=None, help='Ruta del modelo .keras a cargar.')
    parser.add_argument('--iterations', type=int, default=2000, help='Iteraciones para MCTS.')
    parser.add_argument('--depth', type=int, default=3, help='Profundidad para minimax.')
    return parser.parse_args()


def load_model_for_agent(agent, model_path):
    if model_path is None:
        model_path = 'othello_model.keras' if agent == 'mcts' else 'minimax_model.keras'
    return keras.models.load_model(model_path, compile=False)


def choose_ai_move(game, agent, model, iterations, depth, node_counter=None):
    if agent == 'mcts':
        return choose_mcts_move(game, iterations, neural_network=model, node_counter=node_counter)
    return choose_minimax_move(game, depth, neural_network=model, node_counter=node_counter)


def play():
    args = parse_args()
    game = OthelloGame()
    model = load_model_for_agent(args.agent, args.model)
    iterations = args.iterations
    depth = args.depth
    player1 = 1
    player2 = 2
    ia_times = []
    ia_nodes = []

    while not game.is_game_over():
        if game.current_player == player1:
            print_board(game.board)
            print("Player 1 (negro - N)")
            valid_moves = game.get_valid_moves(player1)
            if not valid_moves:
                print("No hay movimientos válidos. Pasando turno.")
                game.current_player = 3 - game.current_player
                continue
                
            print("Movimientos válidos:", valid_moves)
            try:
                move  = random.choice(game.get_valid_moves(player1))
                game.make_move(move, player1)
            except ValueError:
                print("Entrada inválida. Usa números del 0 al 7.")
        else:
            print_board(game.board)
        
        if game.current_player == player2:
            print("Player 2 (blanco - B)")
            print_board(game.board)
            valid_moves = game.get_valid_moves(player2)
            if not valid_moves:
                print("IA no tiene movimientos válidos. Pasando turno.")
                game.current_player = player1
                continue
            start_time = time.time()
            node_counter = [0]
            move = choose_ai_move(game, args.agent, model, iterations, depth, node_counter)
            elapsed = time.time() - start_time  
            ia_times.append(elapsed)  
            ia_nodes.append(node_counter[0])
            print(f"Tiempo de cálculo IA: {elapsed:.3f} segundos")
            print(f"Nodos explorados por la IA: {node_counter[0]}")
            print(move)
            if move is not None:
                game.make_move(move, player2)

        else:
            print_board(game.board)

    print("Juego terminado!")
    print_board(game.board)
    print ("Resutado final:")
    black_count = sum(row.count(1) for row in game.board)
    white_count = sum(row.count(2) for row in game.board)
    print("Fichas negras (N):", black_count)
    print("Fichas blancas (B):", white_count)   
    winner = game.get_winner()
    if winner == 0:
        print("Resultado: Empate")
    elif winner == player2:
        print("Resultado: Gana player 1 (negro): IA")
    else:
        print("Resultado: Gana player 2 (blanco)")

    if ia_times:
        avg_time = sum(ia_times) / len(ia_times)
        print(f"Tiempo medio de cálculo IA: {avg_time:.3f} segundos")
    else:
        print("La IA no realizó ningún movimiento.")

    if ia_nodes:
        total_nodes = sum(ia_nodes)
        print(f"Número total de nodos explorados por la IA: {total_nodes}")
    else:
        print("La IA no exploró ningún nodo.")


if __name__ == "__main__":
    play()
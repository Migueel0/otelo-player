from otelo.otelo import OthelloGame
from minmax.minmax import alphabeta, utility
from tensorflow import keras

import random
import time

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


def play():
    game = OthelloGame()
    model = keras.models.load_model('othello_model.h5') 
    depth = 3 
    player1 = 1
    player2 = 2
    ia_times = []   # Lista para guardar los tiempos de la IA
    ia_nodes = []   # Lista para guardar los nodos explorados por la IA

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
            print("Turno de la IA (blanco - B)...")
            print_board(game.board)
            valid_moves = game.get_valid_moves(player2)
            if not valid_moves:
                print("IA no tiene movimientos válidos. Pasando turno.")
                game.current_player = player1
                continue
            start_time = time.time()  # Inicio medición
            node_counter = [0]
            move, _ = alphabeta(game, depth, float('-inf'), float('inf'), player2, True, model, node_counter)
            elapsed = time.time() - start_time  
            ia_times.append(elapsed)  
            ia_nodes.append(node_counter[0])  # Guardar nodos de este turno
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
    elif winner == player1:
        print("Resultado: Gana player 1 (negro)")
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
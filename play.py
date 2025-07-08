from otelo.otelo import OthelloGame
from minmax.minmax import alphabeta, utility
from tensorflow import keras

import random

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
                row = int(input("Fila (0-7): "))
                col = int(input("Columna (0-7): "))
                if (row, col) not in valid_moves:
                    print("Movimiento inválido. Intenta nuevamente.")
                    continue
                game.make_move((row, col), player1)
                
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
            move, _ = alphabeta(game, depth, float('-inf'), float('inf'), player2, True, model)
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


if __name__ == "__main__":
    play()
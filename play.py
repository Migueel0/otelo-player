
from otelo.otelo import OthelloGame


def print_board(board):
    print("  0 1 2 3 4 5 6 7")
    for i in range(8):
        print(i, end=" ")
        for j in range(8):
            if board[i][j] == 0:
                print(".", end=" ")
            elif board[i][j] == 1:
                print("B", end=" ")
            else:
                print("W", end=" ")
        print()


def play():
    game = OthelloGame()
    player1 = 1
    player2 = 2
    
    while not game.is_game_over():
        
        if game.current_player == player1:
            print_board(game.board)
            print("Player 1 (negro - B)")
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
            print_board(game.board)
            print("Player 2 (blanco - W)")
            valid_moves = game.get_valid_moves(player2)
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
                game.make_move((row, col), player2)
            except ValueError:
                print("Entrada inválida. Usa números del 0 al 7.")
            else:
                print_board(game.board)

    print("Juego terminado!")
    print_board(game.board)
    winner = game.get_winner()
    if winner == 0:
        print("Resultado: Empate")
    elif winner == player1:
        print("Resultado: Gana player 1 (negro)")
    else:
        print("Resultado: Gana player 2 (blanco)")

# Ejecutar el juego
if __name__ == "__main__":
    play()
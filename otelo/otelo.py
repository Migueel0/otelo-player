import copy
class OthelloGame:
    def __init__(self):
        self.board = self.initial_board()
        self.current_player = 1 
        
    def initial_board(self):
        board = [[0]*8 for _ in range(8)]
        board[3][3] = board[4][4] = 2  
        board[3][4] = board[4][3] = 1 
        return board
        
    def get_valid_moves(self, player):
        valid_moves = []
        for i in range(8):
            for j in range(8):
                if self.board[i][j] == 0 and self.is_valid_move(i, j, player):
                    valid_moves.append((i, j))
        return valid_moves
        
    def is_valid_move(self, x, y, player):
        if self.board[x][y] != 0:
            return False
        opponent = 3 - player
        directions = [(-1,-1), (-1,0), (-1,1),
                      (0,-1),         (0,1),
                      (1,-1), (1,0),  (1,1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            found_opponent = False
            while 0 <= nx < 8 and 0 <= ny < 8 and self.board[nx][ny] == opponent:
                nx += dx
                ny += dy
                found_opponent = True
            if found_opponent and 0 <= nx < 8 and 0 <= ny < 8 and self.board[nx][ny] == player:
                return True
        return False
        
    def make_move(self, move, player):
        x, y = move
        if move in self.get_valid_moves(player):
            self.board[x][y] = player
            self.flip_tiles(x, y, player)
            self.current_player = 3 - player  # Cambiar turno
            return True
        return False
        
    def flip_tiles(self, x, y, player):
        opponent = 3 - player
        directions = [(-1,-1), (-1,0), (-1,1),
                      (0,-1),         (0,1),
                      (1,-1), (1,0),  (1,1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            tiles_to_flip = []
            while 0 <= nx < 8 and 0 <= ny < 8 and self.board[nx][ny] == opponent:
                tiles_to_flip.append((nx, ny))
                nx += dx
                ny += dy
            if tiles_to_flip and 0 <= nx < 8 and 0 <= ny < 8 and self.board[nx][ny] == player:
                for fx, fy in tiles_to_flip:
                    self.board[fx][fy] = player
        
    def is_game_over(self):
        return len(self.get_valid_moves(1)) == 0 and len(self.get_valid_moves(2)) == 0
        
    def get_winner(self):
        black = sum(row.count(1) for row in self.board)
        white = sum(row.count(2) for row in self.board)
        if black > white:
            return 1
        elif white > black:
            return 2
        else:
            return 0 

    def copy(self):
        new_game = OthelloGame()
        new_game.board = copy.deepcopy(self.board)
        new_game.current_player = self.current_player
        return new_game
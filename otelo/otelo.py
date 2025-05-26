class OthelloGame:
    def __init__(self):
        self.board = self.initial_board()
        self.current_player = 1 
        
    def initial_board(self):
        board = [[0]*8 for _ in range(8)]
        board[3][3] = board[4][4] = 1  
        board[3][4] = board[4][3] = 2 
        return board
        
    def get_valid_moves(self, player):
        valid_moves = []
        for i in range(8):
            for j in range(8):
                if self.board[i][j] == 0 and self.is_valid_move(i, j, player):
                    valid_moves.append((i, j))
        return valid_moves
        
    def is_valid_move(self, x, y, player):
        directions = [(-1,-1), (-1,0), (-1,1),
                     (0,-1),          (0,1),
                     (1,-1),  (1,0),  (1,1)]
        
        opponent = 3 - player
        valid = False
        
        for dx, dy in directions:
            tx, ty = x + dx, y + dy
            if 0 <= tx < 8 and 0 <= ty < 8 and self.board[tx][ty] == opponent:
                tx += dx
                ty += dy
                while 0 <= tx < 8 and 0 <= ty < 8 and self.board[tx][ty] == opponent:
                    tx += dx
                    ty += dy
                if 0 <= tx < 8 and 0 <= ty < 8 and self.board[tx][ty] == player:
                    valid = True
                    break
        return valid
        
    def make_move(self, move, player):
        x, y = move
        if move in self.get_valid_moves(player):
            self.board[x][y] = player
            self.flip_tiles(x, y, player)
            self.current_player = 3 - player  # Cambiar turno
            return True
        return False
        
    def flip_tiles(self, x, y, player):
        directions = [(-1,-1), (-1,0), (-1,1),
                     (0,-1),          (0,1),
                     (1,-1),  (1,0),  (1,1)]
        opponent = 3 - player
        
        for dx, dy in directions:
            tx, ty = x + dx, y + dy
            to_flip = []
            while 0 <= tx < 8 and 0 <= ty < 8 and self.board[tx][ty] == opponent:
                to_flip.append((tx, ty))
                tx += dx
                ty += dy
            if 0 <= tx < 8 and 0 <= ty < 8 and self.board[tx][ty] == player:
                for (fx, fy) in to_flip:
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
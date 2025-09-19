from tetris import Board, Piece, COLS, ROWS

class Bot_Trainer:
    def __init__(self, game):
        self.game = game
    def evaluate_board(self, board, cleared_rows):
            heights = [0] * COLS
            holes = 0
            bumpiness = 0

            for x in range(COLS):
                col_height = 0
                block_seen = False
                for y in range(ROWS):
                    if (x, y) in board.locked:
                        if not block_seen:
                            col_height = ROWS - y
                            block_seen = True
                    elif block_seen:
                        holes += 1
                heights[x] = col_height

            for i in range(COLS - 1):
                bumpiness += abs(heights[i] - heights[i + 1])
            score = cleared_rows * 10 - 0.5 * sum(heights) - 0.7 * holes - 0.2 * bumpiness
            return score

    def best_move(self):
        best_score = -1e9
        best_piece = None
        for rot in range(len(self.game.current.shape)):
            for x in range(-2, COLS + 2):

                testPiece = Piece(x, self.game.current.y, self.game.current.kind)
                testPiece.rot = rot

                while self.game.board.valid(testPiece):
                    testPiece.y += 1
                testPiece.y -= 1

                if not self.game.board.valid(testPiece):
                    continue

                test_board = Board()
                test_board.locked = self.game.board.locked.copy()
                test_board.add_piece(testPiece)
                cleared = test_board.clear_rows()
                score = self.evaluate_board(test_board, cleared)

                if score > best_score:
                    best_score = score
                    best_piece = testPiece

        return best_piece


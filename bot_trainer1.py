from tetris import Board, Piece, COLS, ROWS

class Bot_Trainer:
    def __init__(self, game, writer=None):
        self.game = game
        self.writer = writer
        self.recorded = False

    def evaluate_board(self, board, cleared_rows):
        heights = [0] * COLS
        holes = 0

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

        bumpiness = sum(abs(heights[i] - heights[i + 1]) for i in range(COLS - 1))
        aggregate_height = sum(heights)
        max_height = max(heights)
        complete_lines = cleared_rows
        holes_density = holes / (COLS * ROWS)
        surface_roughness = bumpiness

        score = cleared_rows * 10 - 0.5 * sum(heights) - 0.7 * holes - 0.2 * bumpiness


        metrics = {
            'heights': heights,
            'holes': holes,
            'bumpiness': bumpiness,
            'cleared': cleared_rows,
            'aggregate_height': aggregate_height,
            'max_height': max_height,
            'complete_lines': complete_lines,
            'holes_density': holes_density,
            'surface_roughness': surface_roughness,
            'score': score
        }

        return score, metrics

    def best_move(self):
        best_score = -1e9
        best_piece = None
        best_metrics = None

        for rot in range(len(self.game.current.shape)):
            for x in range(-2, COLS + 2):
                test_piece = Piece(x, self.game.current.y, self.game.current.kind)
                test_piece.rot = rot

                while self.game.board.valid(test_piece):
                    test_piece.y += 1
                test_piece.y -= 1

                if not self.game.board.valid(test_piece):
                    continue

                test_board = Board()
                test_board.locked = self.game.board.locked.copy()
                test_board.add_piece(test_piece)
                cleared = test_board.clear_rows()
                score, metrics = self.evaluate_board(test_board, cleared)

                if score > best_score:
                    best_score = score
                    best_piece = test_piece
                    best_metrics = metrics
        return best_piece, best_metrics


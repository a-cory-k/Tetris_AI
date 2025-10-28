from tetris_for_gym import Board, Piece, COLS, ROWS
import random

class Bot_Trainer_Random:
    def __init__(self, game, random_every=10):
        self.game = game
        self.random_every = random_every
        self.move_counter = 0
        self.current_target = None
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
    def best_move(self, game=None):
        if game is None:
            game = self.game

        if getattr(game, 'new_piece', True):
            self.current_target = None

        self.move_counter += 1
        use_random = (self.move_counter % self.random_every == 0)

        if use_random:
            if self.current_target is None:
                while True:
                    rand_rot = random.randint(0, len(game.current.shape) - 1)
                    rand_x = random.randint(-2, COLS + 1)
                    piece = Piece(rand_x, game.current.y, game.current.kind)
                    piece.rot = rand_rot
                    temp_board = Board()
                    temp_board.locked = game.board.locked.copy()
                    while temp_board.valid(piece):
                        piece.y += 1
                    piece.y -= 1
                    if temp_board.valid(piece):
                        self.current_target = piece
                        break
            return self.current_target, {'random': True}

        best_score = -1e9
        best_piece = None
        for rot in range(len(game.current.shape)):
            for x in range(-2, COLS + 2):
                test_piece = Piece(x, game.current.y, game.current.kind)
                test_piece.rot = rot
                temp_board = Board()
                temp_board.locked = game.board.locked.copy()
                while temp_board.valid(test_piece):
                    test_piece.y += 1
                test_piece.y -= 1
                if not temp_board.valid(test_piece):
                    continue
                temp_board.add_piece(test_piece)
                cleared = temp_board.clear_rows()
                score, _ = self.evaluate_board(temp_board, cleared)
                if score > best_score:
                    best_score = score
                    best_piece = test_piece
        return best_piece, None

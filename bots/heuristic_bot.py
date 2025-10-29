"""
A heuristic-based bot for playing Tetris.

This module defines the Bot_Trainer class, which uses a set of weighted
heuristics (aggregate height, holes, bumpiness, cleared lines) to
evaluate and choose the best possible move.
"""

from app.tetris_dual import Board, Piece, COLS, ROWS


def apply_move(game, best_piece):
    """
    Applies the selected best move to the game state.

    If best_piece is not None, it sets the game's current piece
    to the final position and rotation and performs a hard drop.
    If best_piece is None, it just hard drops the current piece as a fallback.

    Args:
        game: The main game state object.
        best_piece (Piece or None): The calculated best piece placement.
                                 This piece is assumed to be in its
                                 final (lowest) valid position.
    """
    if best_piece:
        final_move_piece = Piece(best_piece.x, game.current.y, best_piece.kind)
        final_move_piece.rot = best_piece.rot
        game.current = final_move_piece
        game.hard_drop()
    else:
        game.hard_drop()

class Bot_Trainer:
    """
    Implements a heuristic Tetris bot that finds the best move.

    This bot simulates all possible piece placements (rotations and x-positions)
    and evaluates the resulting board state using a linear combination of
    features.
    """
    def __init__(self, writer=None):
        """
        Initializes the bot.

        Args:
            writer (SummaryWriter, optional): A TensorBoard writer for logging.
        """
        self.writer = writer
        self.recorded = False

    # pylint: disable=too-many-locals
    def evaluate_board(self, board, cleared_rows):
        """
        Calculates a score for a given board state based on heuristics.

        Args:
            board (Board): The board state to evaluate.
            cleared_rows (int): The number of rows cleared in this move.

        Returns:
            tuple: A tuple containing:
                - score (float): The calculated heuristic score.
                - metrics (dict): A dictionary of the individual heuristic values.
        """
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

        # weights
        a = -0.5  # height
        b = 0.7  # rows
        c = -0.3   # holes
        d = -0.2  # bumpiness

        score = (a * aggregate_height) + (b * cleared_rows) + (c * holes) + (d * bumpiness)

        metrics = {
            'heights': heights,
            'holes': holes,
            'bumpiness': bumpiness,
            'cleared': cleared_rows,
            'aggregate_height': aggregate_height,
            'score': score
        }
        return score, metrics


    def best_move(self, game):
        """
        Finds and executes the best move for the current piece.

        It iterates through all possible rotations and horizontal positions,
        simulates the move, and scores the resulting board.

        Args:
            game: The main game state object.

        Returns:
            tuple: A tuple containing (None, best_metrics) where best_metrics
                   is a dict of the heuristics for the chosen move.
        """
        best_score = -1e9
        best_piece = None
        best_metrics = None

        current_piece = game.current

        for rot in range(len(current_piece.shape)):
            for x in range(-2, COLS + 2):
                test_piece = Piece(x, current_piece.y, current_piece.kind)
                test_piece.rot = rot

                if not game.board.valid(test_piece):
                    continue

                while True:
                    test_piece.y += 1
                    if not game.board.valid(test_piece):
                        test_piece.y -= 1
                        break

                test_board = Board()
                test_board.locked = game.board.locked.copy()
                test_board.add_piece(test_piece)
                cleared = test_board.clear_rows()

                if test_board.check_lost():
                    continue
                score, metrics = self.evaluate_board(test_board, cleared)
                if score > best_score:
                    best_score = score
                    best_piece = test_piece
                    best_metrics = metrics

        apply_move(game, best_piece)

        return None, best_metrics

"""
Module containing the Bot_Trainer_Random class for Tetris.
This bot alternates between calculated and random moves for training.
"""

#import random
from app.tetris_dual import Board, Piece, COLS
from bots import heuristic_bot

# pylint: disable=too-few-public-methods
class Bot_Trainer_Random:
    """
    Implements a Tetris bot that calculates the best move,
    but periodically (every 'random_every' move) makes a random move
    for exploration.
    """

    def __init__(self, game, random_every=10):
        """
        Initializes the bot.

        Args:
            game: The current game object.
            random_every (int): How often to make a random move (every Nth move).
        """
        self.game = game
        self.random_every = random_every
        self.move_counter = 0
        self.current_target = None

        self.eval_bot = heuristic_bot.Bot_Trainer(game)
    def best_move(self, game=None):
        """
        Finds the best move or returns a random one.

        Every 'random_every' move, a random valid move will be chosen.
        Otherwise, it will brute-force all possible moves
        to find the best one according to 'evaluate_board'.

        Args:
            game: The game object (uses self.game if None).

        Returns:
            tuple: (Piece target_piece, dict info)
                   target_piece - The piece object in its final position.
                   info - A dict (None, or {'random': True} for a random move).
        """
        if game is None:
            game = self.game

        if getattr(game, 'new_piece', True):
            self.current_target = None

        self.move_counter += 1
        #use_random = self.move_counter % self.random_every == 0

        #if use_random:
        #    if self.current_target is None:
        #        while True:
        #            rand_rot = random.randint(0, len(game.current.shape) - 1)
        #            rand_x = random.randint(-2, COLS + 1)
        #            piece = Piece(rand_x, game.current.y, game.current.kind)
        #            piece.rot = rand_rot
        #            temp_board = Board()
        #            temp_board.locked = game.board.locked.copy()
        #            while temp_board.valid(piece):
        #                piece.y += 1
        #            piece.y -= 1
        #            if temp_board.valid(piece):
        #                self.current_target = piece
        #                break
        #    return self.current_target, {'random': True}

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
                score, _ = self.eval_bot.evaluate_board(temp_board, cleared)
                if score > best_score:
                    best_score = score
                    best_piece = test_piece
        return best_piece, None

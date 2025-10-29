"""
Module for collecting Tetris game data.

This module defines a class 'TetrisDataCollector' that runs Tetris games
using a bot and records various game states and bot decisions to a CSV file
for dataset creation.
"""

import csv
import numpy as np
import pygame
from app.tetris_dual import App, Piece, Board, COLS
from bots.heuristic_bot import Bot_Trainer


class TetrisDataCollector:
    """
    Runs automated Tetris games to collect data for machine learning.

    Can be configured to save only the bot's best move or all possible
    moves with their scores.
    """

    def __init__(self, filename="tetris_dataset.csv", num_games=100,
                 fps_limit=None, mode="best"):
        """
        Initializes the data collector.

        Args:
            filename (str): The name of the output CSV file.
            num_games (int): The number of games to simulate.
            fps_limit (int, optional): If set, limits the game speed to
                                       this FPS. Defaults to None (unlimited).
            mode (str): "best" (only record the chosen move) or
                        "all" (record all possible moves).
        """
        self.filename = filename
        self.num_games = num_games
        self.fps_limit = fps_limit
        self.mode = mode

    def run(self):
        """
        Runs the full data collection process for the specified number of games.
        """
        # W1514: Added encoding="utf-8"
        with open(self.filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                'game_id', 'kind', 'rot', 'x', 'heights', 'holes', 'bumpiness',
                'cleared', 'aggregate_height', 'max_height', 'holes_density',
                'surface_roughness', 'score', 'is_best'
            ])
            for game_id in range(1, self.num_games + 1):
                print(f"Game {game_id}/{self.num_games}")
                self.run_single_game(writer, game_id)

    def run_single_game(self, writer, game_id):
        """
        Runs a single game of Tetris and writes data to the CSV writer.

        Args:
            writer: The csv.writer object.
            game_id (int): The identifier for the current game.
        """
        app = App()
        game = app.game
        bot = Bot_Trainer(game)

        clock = None
        if self.fps_limit:
            clock = pygame.time.Clock()

        running = True
        while running:
            if self.fps_limit:
                clock.tick(self.fps_limit)
            if game.over:
                running = False
                continue

            if self.mode == "best":
                self.record_best_move(writer, bot, game, game_id)
            elif self.mode == "all":
                self.record_all_moves(writer, bot, game, game_id)

    def _create_metrics_row(self, game_id, piece, metrics, is_best):
        """Helper function to create a standardized data row."""
        return [
            game_id,
            piece.kind,
            piece.rot,
            piece.x,
            ",".join(map(str, metrics['heights'])),
            metrics['holes'],
            metrics['bumpiness'],
            metrics['cleared'],
            metrics['aggregate_height'],
            metrics['max_height'],
            metrics['holes_density'],
            metrics['surface_roughness'],
            metrics['score'],
            1 if is_best else 0
        ]

    def record_best_move(self, writer, bot, game, game_id):
        """
        Finds the best move, records it, and advances the game.

        Args:
            writer: The csv.writer object.
            bot (Bot_Trainer): The bot instance.
            game (Game): The current game state.
            game_id (int): The identifier for the current game.
        """
        # E1120: Pass 'game' to 'best_move'
        best_piece, metrics = bot.best_move(game)
        game.current = best_piece
        game.lock_piece()

        row = self._create_metrics_row(game_id, best_piece, metrics, True)
        writer.writerow(row)
    # pylint:disable = R0914
    def record_all_moves(self, writer, bot, game, game_id):
        """
        Records all possible moves for the current piece and advances the game.

        Args:
            writer: The csv.writer object.
            bot (Bot_Trainer): The bot instance.
            game (Game): The current game state.
            game_id (int): The identifier for the current game.
        """
        current_kind = game.current.kind
        current_y = game.current.y
        shape_len = len(game.current.shape)
        best_piece, best_metrics = bot.best_move(game)
        row = self._create_metrics_row(game_id, best_piece, best_metrics, True)
        writer.writerow(row)

        for rot in range(shape_len):
            for x in range(-2, COLS + 2):
                piece = Piece(x, current_y, current_kind)
                piece.rot = rot

                temp_board = Board()
                temp_board.locked = game.board.locked.copy()

                while temp_board.valid(piece):
                    piece.y += 1
                piece.y -= 1

                if not temp_board.valid(piece):
                    continue

                if (piece.kind == best_piece.kind and
                        piece.rot == best_piece.rot and
                        piece.x == best_piece.x):
                    continue

                temp_board.add_piece(piece)
                cleared = temp_board.clear_rows()
                _, metrics = bot.evaluate_board(temp_board, cleared)

                row = self._create_metrics_row(game_id, piece, metrics, False)
                writer.writerow(row)

        game.current = best_piece
        game.lock_piece()

    def run_cnn_best(self, filename="tetris_dataset_cnn_best.csv"):
        """
        Runs data collection for CNN, saving the board state.

        Args:
            filename (str): Output CSV file name.
        """
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ['game_id'] + [f'c{i}' for i in range(20 * 10)] + \
                     ['kind', 'rot', 'x', 'action']
            writer.writerow(header)

            for game_id in range(1, self.num_games + 1):
                print(f"Game {game_id}/{self.num_games}")
                self.run_single_game_cnn(writer, game_id)

    def run_single_game_cnn(self, writer, game_id):
        """
        Runs a single game and records board state data for CNN.

        Args:
            writer: The csv.writer object.
            game_id (int): The identifier for the current game.
        """
        app = App()
        game = app.game
        bot = Bot_Trainer(game)

        clock = None
        if self.fps_limit:
            clock = pygame.time.Clock()

        running = True
        while running:
            if self.fps_limit:
                clock.tick(self.fps_limit)
            if game.over:
                running = False
                continue

            best_piece, _ = bot.best_move(game)
            game.current = best_piece
            game.lock_piece()
            self.record_cnn_data(writer, game_id, game, best_piece, 'best')

    # pylint: disable=R0913
    # pylint: disable=R0917
    def record_cnn_data(self, writer, game_id, game, piece, action):
        """
        Writes a single row of CNN-formatted data.

        Args:
            writer: The csv.writer object.
            game_id (int): The identifier for the current game.
            game (Game): The current game state.
            piece (Piece): The piece that was placed.
            action (str): A label for the action (e.g., 'best').
        """
        field_flat = self.get_board_matrix(game.board).flatten()
        x_val = piece.x
        row = [game_id] + field_flat.tolist() + \
              [piece.kind, piece.rot, x_val, action]
        writer.writerow(row)

    def get_board_matrix(self, board, rows=20, cols=10):
        """
        Converts the board's locked pieces into a 2D numpy array.

        Args:
            board (Board): The game board.
            rows (int): The height of the matrix.
            cols (int): The width of the matrix.

        Returns:
            np.array: A 2D array (matrix) representing the board.
        """
        mat = np.zeros((rows, cols), dtype=int)
        for (x, y), _ in board.locked.items():
            if 0 <= y < rows and 0 <= x < cols:
                mat[y, x] = 1
        return mat


if __name__ == "__main__":
    collector = TetrisDataCollector(
        filename="tetris_dataset_only_good_moves.csv",
        num_games=100,
        fps_limit=None,  # Set to 60 to limit FPS, None for max speed
        mode="best"
    )
    # collector.run()
    collector.run_cnn_best(filename="tetris_dataset_cnn_best.csv")

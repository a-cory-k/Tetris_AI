import csv
import numpy as np
from tetris import App, Piece, Board, COLS
from bot_trainer1 import Bot_Trainer

class TetrisDataCollector:
    def __init__(self, filename="tetris_dataset_only_good_moves.csv", num_games=100, limit_fps=False, fps=60, mode="best"):
        self.filename = filename
        self.num_games = num_games
        self.limit_fps = limit_fps
        self.fps = fps
        self.mode = mode

    # ---------------------- OLD CSV ----------------------
    def run(self):
        with open(self.filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                'game_id','kind','rot','x','heights','holes','bumpiness',
                'cleared','aggregate_height','max_height','holes_density',
                'surface_roughness','score','is_best'
            ])
            for game_id in range(1, self.num_games + 1):
                print(f"Game {game_id}/{self.num_games}")
                self.run_single_game(writer, game_id)

    def run_single_game(self, writer, game_id):
        app = App()
        game = app.game
        bot = Bot_Trainer(game)

        clock = None
        if self.limit_fps:
            import pygame
            clock = pygame.time.Clock()

        running = True
        while running:
            if self.limit_fps:
                clock.tick(self.fps)
            if game.over:
                running = False
                continue

            if self.mode == "best":
                self.record_best_move(writer, bot, game, game_id)
            elif self.mode == "all":
                self.record_all_moves(writer, bot, game, game_id)

    def record_best_move(self, writer, bot, game, game_id):
        best_piece, metrics = bot.best_move()
        game.current = best_piece
        game.lock_piece()

        row = [
            game_id,
            best_piece.kind,
            best_piece.rot,
            best_piece.x,
            ",".join(map(str, metrics['heights'])),
            metrics['holes'],
            metrics['bumpiness'],
            metrics['cleared'],
            metrics['aggregate_height'],
            metrics['max_height'],
            metrics['holes_density'],
            metrics['surface_roughness'],
            metrics['score'],
            1
        ]
        writer.writerow(row)

    def record_all_moves(self, writer, bot, game, game_id):
        current_kind = game.current.kind
        current_y = game.current.y
        shape_len = len(game.current.shape)

        best_piece, best_metrics = bot.best_move()
        game.current = best_piece
        game.lock_piece()

        row = [
            game_id,
            best_piece.kind,
            best_piece.rot,
            best_piece.x,
            ",".join(map(str, best_metrics['heights'])),
            best_metrics['holes'],
            best_metrics['bumpiness'],
            best_metrics['cleared'],
            best_metrics['score'],
            1
        ]
        writer.writerow(row)

        for rot in range(shape_len):
            for x in range(-2, COLS + 2):
                piece = Piece(x, current_y, current_kind)
                piece.rot = rot

                while game.board.valid(piece):
                    piece.y += 1
                piece.y -= 1

                if not game.board.valid(piece):
                    continue

                temp_board = Board()
                temp_board.locked = game.board.locked.copy()
                temp_board.add_piece(piece)
                cleared = temp_board.clear_rows()
                score, metrics = bot.evaluate_board(temp_board, cleared)

                if (piece.kind == best_piece.kind and
                    piece.rot == best_piece.rot and
                    piece.x == best_piece.x):
                    continue

                row = [
                    game_id,
                    piece.kind,
                    piece.rot,
                    piece.x,
                    ",".join(map(str, metrics['heights'])),
                    metrics['holes'],
                    metrics['bumpiness'],
                    metrics['cleared'],
                    metrics['score'],
                    0
                ]
                writer.writerow(row)

    # ---------------------- CNN CSV ----------------------
    def run_cnn_best(self, filename="tetris_dataset_cnn_best.csv"):
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            header = ['game_id'] + [f'c{i}' for i in range(20*10)] + ['kind','rot','x','action']
            writer.writerow(header)

            for game_id in range(1, self.num_games + 1):
                print(f"Game {game_id}/{self.num_games}")
                self.run_single_game_cnn(writer, game_id)

    def run_single_game_cnn(self, writer, game_id):
        app = App()
        game = app.game
        bot = Bot_Trainer(game)

        clock = None
        if self.limit_fps:
            import pygame
            clock = pygame.time.Clock()

        running = True
        while running:
            if self.limit_fps:
                clock.tick(self.fps)
            if game.over:
                running = False
                continue

            best_piece, metrics = bot.best_move()
            game.current = best_piece
            game.lock_piece()
            self.record_cnn_data(writer, game_id, game, best_piece, action='best')

    def record_cnn_data(self, writer, game_id, game, piece, action):
        field_flat = self.get_board_matrix(game.board).flatten()
        x_val = piece.x
        if x_val < 0:
            print(f"Game {game_id}, piece {piece.kind}, rot {piece.rot}, x={x_val}, y={piece.y}")
        row = [game_id] + field_flat.tolist() + [piece.kind, piece.rot, x_val, action]
        writer.writerow(row)

    def get_board_matrix(self, board, rows=20, cols=10):
        mat = np.zeros((rows, cols), dtype=int)
        for (x, y), val in board.locked.items():
            if 0 <= y < rows and 0 <= x < cols:
                mat[y, x] = 1
        return mat


if __name__ == "__main__":
    collector = TetrisDataCollector(
        filename="tetris_dataset_only_good_moves.csv",
        num_games=100,
        limit_fps=False,
        fps=60,
        mode="best"
    )


    #collector.run()
    collector.run_cnn_best(filename="tetris_dataset_cnn_best.csv")

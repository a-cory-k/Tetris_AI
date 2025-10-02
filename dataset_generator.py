import csv
from tetris import App, Piece, Board, COLS
from bot_trainer1 import Bot_Trainer

class TetrisDataCollector:
    def __init__(self, filename="dataset.csv", num_games=200, limit_fps=False, fps=60, mode="best"):
        self.filename = filename
        self.num_games = num_games
        self.limit_fps = limit_fps
        self.fps = fps
        self.mode = mode

    def run(self):
        with open(self.filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                'game_id', 'kind', 'rot', 'x',
                'heights', 'holes', 'bumpiness',
                'cleared', 'score', 'is_best'
            ])

            for game_id in range(1, self.num_games + 1):
                print(f"Game {game_id}/{self.num_games}")
                self._run_single_game(writer, game_id)

    def _run_single_game(self, writer, game_id):
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
                self._record_best_move(writer, bot, game, game_id)
            elif self.mode == "all":
                self._record_all_moves(writer, bot, game, game_id)

    def _record_best_move(self, writer, bot, game, game_id):
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
            metrics['score'],
            1
        ]
        writer.writerow(row)



    def _record_all_moves(self, writer, bot, game, game_id):
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

                if (
                        piece.kind == best_piece.kind and
                        piece.rot == best_piece.rot and
                        piece.x == best_piece.x
                ):
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


if __name__ == "__main__":
    collector = TetrisDataCollector(
        filename="tetris_dataset_all_moves.csv",
        num_games=200,
        limit_fps=False,
        fps=60,
        mode="all" # if want write only good moves use best, else use all
    )
    collector.run()


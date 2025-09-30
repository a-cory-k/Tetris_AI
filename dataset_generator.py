import csv
from tetris import App
from bot_trainer1 import Bot_Trainer
import pygame

with open("tetris_dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['game_id','kind','rot','x','heights','holes','bumpiness','cleared','score'])
    cur_game = 0
    while cur_game != 200:
        app = App()
        game = app.game
        bot = Bot_Trainer(game)

        clock = pygame.time.Clock()
        running = True
        cur_game += 1
        print(cur_game)
        while running:
            dt = clock.tick(60)/1000.0
            if game.over:
                running = False
                continue

            # best_move
            best_piece, metrics = bot.best_move()
            game.current = best_piece
            game.lock_piece()

            # write metrics into CSV after lock
            row = [
                cur_game,
                best_piece.kind,
                best_piece.rot,
                best_piece.x,
                ",".join(map(str, metrics['heights'])),
                metrics['holes'],
                metrics['bumpiness'],
                metrics['cleared'],
                metrics['score']
            ]
            writer.writerow(row)


import pygame
from app.tetris_dual import AppDual, Game
from bots.heuristic_bot import Bot_Trainer
#from tetris_train_not_olny_good import Bot_Trainer_Norm
#from trainer_random import Bot_Trainer_Random
#from bot_dqn import DQNBot
from bots.cnn_bot import CnnBot
from pathlib import Path
from bots.dqn_bot_CNN import DQNBot
pygame.init()
screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption("TETRIS MENU")
font = pygame.font.SysFont("consolas", 36, bold=True)
small_font = pygame.font.SysFont("consolas", 24)

WHITE = (255, 255, 255)
GRAY = (180, 180, 180)
BG = (20, 20, 30)

def draw_text_center(text, y, color=WHITE):
    label = font.render(text, True, color)
    x = 300 - label.get_width() // 2
    screen.blit(label, (x, y))

def menu(options, title="TETRIS"):
    selected = 0
    while True:
        screen.fill(BG)
        draw_text_center(title, 50)
        for i, option in enumerate(options):
            color = WHITE if i == selected else GRAY
            label = small_font.render(option, True, color)
            x = 300 - label.get_width() // 2
            y = 150 + i * 50
            screen.blit(label, (x, y))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected = (selected - 1) % len(options)
                elif event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(options)
                elif event.key == pygame.K_RETURN:
                    return options[selected]


BOT_CLASSES = {
    "Heuristic Bot": Bot_Trainer,
    #"Normalized Bot": Bot_Trainer_Norm,
    #"Random Bot": Bot_Trainer_Random,
    "DQN CNN Bot v.1": lambda g: DQNBot(str(Path(__file__).resolve().parent / "models" / "tetris_dqn_v1.pth")),
    "DQN CNN Bot v.2 (Better)": lambda g: DQNBot(str(Path(__file__).resolve().parent / "models" / "tetris_dqn_v2.pth")),
    "CNN Bot": lambda g: CnnBot(str(Path(__file__).resolve().parent / "models" / "tetris_cnn.pth"))
}

def choose_bot():
    choice = menu(list(BOT_CLASSES.keys()), title="Choose Bot")
    if choice is None:
        return None
    return BOT_CLASSES[choice]

def start_game(mode):
    if mode == "Player vs Player":
        app = AppDual(bot_enabled=False)
        app.run()
        return

    elif mode == "Player vs Bot":
        BotClass = choose_bot()
        if BotClass is None:
            return

        game_for_bot = Game()
        bot = BotClass(game_for_bot) if callable(BotClass) else BotClass
        app = AppDual(bot_enabled=True, bot=bot)
        app.games[1] = game_for_bot
        app.run()
        return

    elif mode == "Bot vs Bot":
        BotClass1 = choose_bot()
        if BotClass1 is None:
            return
        BotClass2 = choose_bot()
        if BotClass2 is None:
            return

        g1, g2 = Game(), Game()
        bot1 = BotClass1(g1) if callable(BotClass1) else BotClass1
        bot2 = BotClass2(g2) if callable(BotClass2) else BotClass2

        app = AppDual(bot_enabled=True)
        app.games = [g1, g2]
        app.bot1 = bot1
        app.bot2 = bot2
        app.run()
        return

MAIN_OPTIONS = ["Player vs Player", "Bot vs Bot", "Player vs Bot", "Exit"]

while True:
    choice = menu(MAIN_OPTIONS)
    if not choice or choice == "Exit":
        pygame.quit()
        break
    start_game(choice)


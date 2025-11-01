# pylint: disable=no-member
"""
Main menu and entry point for the Tetris AI application.

This script initializes Pygame and displays a main menu that allows the
user to select different game modes, such as Single Player (Human or Bot),
Player vs. Player, Bot vs. Bot, and Player vs. Bot.
"""

from pathlib import Path
import pygame
import numpy as np  
from app.tetris_dual import App, AppDual, Game
from bots.heuristic_bot import Bot_Trainer
from bots.cnn_bot import CnnBot
from bots.dqn_bot_CNN import DQNBot

from bots.lstm_bot import LstmBot

pygame.init()
screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption("TETRIS MENU")

font = pygame.font.SysFont("consolas", 40, bold=True)
small_font = pygame.font.SysFont("consolas", 28)

BG = (20, 20, 30)
WHITE = (255, 255, 255)
GRAY = (100, 100, 110)
NEON_CYAN = (0, 255, 255)


def draw_text_center(text, y, color=WHITE, font_obj=None):
    """
    Renders and blits text centered horizontally on the screen.

    Args:
        text (str): The text to render.
        y (int): The y-coordinate for the text.
        color (tuple, optional): The RGB color of the text. Defaults to WHITE.
        font_obj (pygame.font.Font, optional): The font to use. Defaults to global 'font'.
    """
    current_font = font_obj if font_obj else font

    label = current_font.render(text, True, color)
    x = 300 - label.get_width() // 2
    screen.blit(label, (x, y))


def menu(options, title="TETRIS"):
    """
    Displays an interactive, selectable retro-style menu and returns the chosen option.

    Args:
        options (list): A list of strings for the menu options.
        title (str, optional): The title to display above the menu.

    Returns:
        str or None: The string of the selected option, or None if the
                     user quits.
    """
    selected = 0
    clock = pygame.time.Clock()
    blink_timer = 0
    show_selector = True

    while True:
        dt = clock.tick(60)
        blink_timer += dt
        if blink_timer >= 400:
            blink_timer %= 400
            show_selector = not show_selector

        screen.fill(BG)

        draw_text_center(title, 50, WHITE)
        pygame.draw.line(screen, NEON_CYAN, (100, 90), (500, 90), 2)
        pygame.draw.line(screen, NEON_CYAN, (120, 100), (480, 100), 1)

        for i, option in enumerate(options):
            y = 150 + i * 50
            if i == selected:
                color = NEON_CYAN

                text = f"  {option}  "
                if show_selector:
                    text = f"> {option} <"

                draw_text_center(text, y, color, small_font)

                if show_selector:
                    glow_color = (NEON_CYAN[0] // 4, NEON_CYAN[1] // 4, NEON_CYAN[2] // 4)
                    label_w = small_font.render(text, True, color).get_width()
                    line_x_start = 300 - label_w // 2 - 15
                    line_x_end = 300 + label_w // 2 + 15
                    pygame.draw.line(screen, glow_color, (line_x_start, y + 12), (line_x_start + 10, y + 12), 2)
                    pygame.draw.line(screen, glow_color, (line_x_end - 10, y + 12), (line_x_end, y + 12), 2)

            else:
                color = GRAY
                text = f"  {option}  "
                draw_text_center(text, y, color, small_font)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected = (selected - 1) % len(options)
                elif event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(options)
                elif event.key == pygame.K_RETURN:
                    return options[selected]

MODEL_DIR = Path(__file__).resolve().parent / "models"
SCALER_PATH = MODEL_DIR / "scaler_params.npz"

try:
    scaler_params = np.load(str(SCALER_PATH))
    LSTM_SCALER_MEAN = scaler_params['mean']
    LSTM_SCALER_SCALE = scaler_params['scale']
    LSTM_BOT_READY = True
except FileNotFoundError:
    print(f"WARNING: LstmBot scaler file not found at {SCALER_PATH}")
    print("LSTM Bot will be disabled.")
    LSTM_BOT_READY = False

BOT_CLASSES = {
    "Heuristic Bot": Bot_Trainer,
    "DQN CNN Bot v.1": lambda g: DQNBot(str(MODEL_DIR / "tetris_dqn_v1.pth")),
    "DQN CNN Bot v.2 (Better)": lambda g: DQNBot(str(MODEL_DIR / "tetris_dqn_v2.pth")),
    "CNN Bot": lambda g: CnnBot(str(MODEL_DIR / "tetris_cnn.pth"))
}

if LSTM_BOT_READY:
    BOT_CLASSES["LSTM Bot"] = lambda g: LstmBot(
        model_path=str(MODEL_DIR / "tetris_actor_lstm_best.pth"),
        scaler_mean=LSTM_SCALER_MEAN,
        scaler_scale=LSTM_SCALER_SCALE
    )

def choose_bot():
    """
    Displays a menu for the user to select a bot.

    Returns:
        A bot class or factory function from BOT_CLASSES, or None.
    """
    bot_options = sorted(list(BOT_CLASSES.keys()))

    bot_choice = menu(bot_options, title="Choose Bot")
    if bot_choice is None:
        return None
    return BOT_CLASSES[bot_choice]


# pylint: disable=too-many-return-statements
def start_game(mode):
    """
    Launches the correct game mode based on the user's menu selection.

    Args:
        mode (str): The string corresponding to the user's choice.
    """
    if mode == "Single Player":
        single_player_choice = menu(["Human Player", "Bot Player"], title="Single Player Mode")
        if single_player_choice is None:
            return

        if single_player_choice == "Human Player":
            app = App()
            app.run(bot_enabled=False)
            return

        if single_player_choice == "Bot Player":
            BotClass = choose_bot()
            if BotClass is None:
                return

            game_for_bot = Game()
            bot = BotClass(game_for_bot) if callable(BotClass) else BotClass

            app = App()
            app.game = game_for_bot
            app.run(bot_enabled=True, bot=bot)
            return
    elif mode == "Player vs Player":
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


MAIN_OPTIONS = ["Single Player", "Player vs Player", "Bot vs Bot", "Player vs Bot", "Exit"]

if __name__ == "__main__":
    while True:
        choice = menu(MAIN_OPTIONS)
        if not choice or choice == "Exit":
            pygame.quit()
            break
        start_game(choice)

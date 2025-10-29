# pylint: disable=no-member
"""
A dual-screen and single-screen Tetris game implementation using Pygame.

This module defines all the core logic for a Tetris game, including:
- Piece and Board classes
- Game logic for single-player
- Renderer class for drawing
- App class for single-player game loop
- AppDual class for a two-player, side-by-side game loop
"""

import random
import pygame


COLS, ROWS = 10, 20
BLOCK = 32
PLAY_W, PLAY_H = COLS * BLOCK, ROWS * BLOCK
SIDE, MARGIN = 200, 20
WIDTH, HEIGHT = PLAY_W + SIDE + MARGIN * 3, PLAY_H + MARGIN * 2
FPS = 60
BASE_FALL_TIME = 0.75

WHITE = (240, 240, 240)
LIGHT = (210, 210, 210)
DARK = (30, 30, 30)
BG = (15, 15, 20)
GRID = (45, 45, 55)

COLORS = [
    (80, 190, 230),  # I
    (240, 240, 90),  # O
    (240, 150, 80),  # L
    (100, 180, 90),  # S
    (200, 100, 200),  # T
    (70, 90, 200),  # J
    (240, 90, 100)  # Z
]

SHAPES = {
    'I': [
        ["....",
         "####",
         "....",
         "...."],
        ["..#.",
         "..#.",
         "..#.",
         "..#."]
    ],
    'O': [
        [".##.",
         ".##.",
         "....",
         "...."]
    ],
    'T': [
        [".#..",
         "###.",
         "....",
         "...."],
        [".#..",
         ".##.",
         ".#..",
         "...."],
        ["....",
         "###.",
         ".#..",
         "...."],
        [".#..",
         "##..",
         ".#..",
         "...."]
    ],
    'S': [
        ["..##",
         ".##.",
         "....",
         "...."],
        [".#..",
         ".##.",
         "..#.",
         "...."]
    ],
    'Z': [
        [".##.",
         "..##",
         "....",
         "...."],
        ["..#.",
         ".##.",
         ".#..",
         "...."]
    ],
    'J': [
        ["#...",
         "###.",
         "....",
         "...."],
        [".##.",
         ".#..",
         ".#..",
         "...."],
        ["....",
         "###.",
         "..#.",
         "...."],
        [".#..",
         ".#..",
         "##..",
         "...."]
    ],
    'L': [
        ["..#.",
         "###.",
         "....",
         "...."],
        [".#..",
         ".#..",
         ".##.",
         "...."],
        ["....",
         "###.",
         "#...",
         "...."],
        ["##..",
         ".#..",
         ".#..",
         "...."]
    ]
}

ORDER = ['I', 'O', 'L', 'S', 'T', 'J', 'Z']


class Piece:
    """Represents a single Tetris piece (tetromino)."""

    def __init__(self, x, y, kind):
        """
        Initializes a new Piece.

        Args:
            x (int): The initial x-coordinate (grid-based).
            y (int): The initial y-coordinate (grid-based).
            kind (str): The shape type ('I', 'O', 'T', etc.).
        """
        self.x = x
        self.y = y
        self.kind = kind
        self.rot = 0
        self.shape = SHAPES[kind]
        self.color = COLORS[ORDER.index(kind)]

    def cells(self):
        """
        Calculates the 4 (x, y) grid coordinates of the piece's blocks
        based on its current position and rotation.
        """
        pattern = self.shape[self.rot]
        positions = []
        x_offset, y_offset = 1, 1
        if self.kind == 'I' and self.rot == 1:
            x_offset = 2
        for r_idx, row in enumerate(pattern):
            for c_idx, cell in enumerate(row):
                if cell == '#':
                    positions.append((int(self.x + c_idx - x_offset),
                                      int(self.y + r_idx - y_offset)))
        return positions

    def rotated(self, direction=1):
        """
        Creates a new Piece instance that is a rotated version of this one.

        Args:
            direction (int): 1 for clockwise (default), -1 for counter-clockwise.
        """
        p = Piece(self.x, self.y, self.kind)
        p.rot = (self.rot + direction) % len(self.shape)
        return p


class Board:
    """Represents the game board and all locked pieces."""

    def __init__(self):
        """Initializes an empty board."""
        self.locked = {}
        self.cleared_lines = 0

    def create_grid(self):
        """Creates a 2D list representation of the board for rendering."""
        grid = [[None for _ in range(COLS)] for _ in range(ROWS)]
        for (x, y), color in self.locked.items():
            if 0 <= x < COLS and 0 <= y < ROWS:
                grid[y][x] = color
        return grid

    def valid(self, piece):
        """Checks if a piece's position is valid (on-grid, not overlapping)."""
        for x, y in piece.cells():
            if x < 0 or x >= COLS or y >= ROWS:
                return False
            if y >= 0 and (x, y) in self.locked:
                return False
        return True

    def add_piece(self, piece):
        """Locks a piece onto the board."""
        for x, y in piece.cells():
            self.locked[(x, y)] = piece.color

    def check_lost(self):
        """Checks if any locked piece is above the visible playfield."""
        return any(y < 0 for _, y in self.locked)

    def clear_rows(self):
        """Checks for and clears completed rows, shifting pieces down."""
        full_rows = [y for y in range(ROWS)
                     if all((x, y) in self.locked for x in range(COLS))]
        if not full_rows:
            return 0

        for y in full_rows:
            for x in range(COLS):
                self.locked.pop((x, y))

        new_locked = {}
        for (x, y), color in self.locked.items():
            shift = sum(1 for ry in full_rows if ry > y)
            new_locked[(x, y + shift)] = color
        self.locked = new_locked
        self.cleared_lines += len(full_rows)
        return len(full_rows)


# pylint: disable=too-many-instance-attributes
class Game:
    """Manages the state of a single Tetris game."""

    def __init__(self):
        """Initializes a new game session."""
        self.board = Board()
        self.queue = self.new_bag()
        self.current = self.get_next()
        self.next_piece = self.peek_next()
        self.score = self.lines = 0
        self.fall_time, self.fall_interval = 0.0, BASE_FALL_TIME
        self.soft_drop = False
        self.paused = False
        self.over = False

    def new_bag(self):
        """Creates a new 'bag' of all 7 pieces, shuffled."""
        bag = ORDER.copy()
        random.shuffle(bag)
        return bag

    def get_next(self):
        """Gets the next piece from the queue, refilling if needed."""
        while len(self.queue) < 3:
            self.queue.extend(self.new_bag())
        return Piece(COLS // 2, -1, self.queue.pop(0))

    def peek_next(self, n=3):
        """Looks at the next N pieces in the queue without removing them."""
        while len(self.queue) < n:
            self.queue.extend(self.new_bag())
        return self.queue[:n]

    def hard_drop(self):
        """Instantly drops the current piece to the bottom."""
        while True:
            m = Piece(self.current.x, self.current.y + 1, self.current.kind)
            m.rot = self.current.rot
            if self.board.valid(m):
                self.current = m
            else:
                break
        self.lock_piece()

    def lock_piece(self):
        """Locks the current piece, checks lines, and gets the next piece."""
        self.board.add_piece(self.current)
        cleared = self.board.clear_rows()
        if cleared:
            self.lines += cleared
            self.score += {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}[cleared]
        self.current = self.get_next()
        self.next_piece = self.peek_next()
        if not self.board.valid(self.current) or self.board.check_lost():
            self.over = True

    def step(self, dt):
        """
        Advances the game state by one time step (dt).
        Handles gravity and piece locking.
        """
        if self.paused or self.over:
            return
        self.fall_time += dt
        speed = self.fall_interval * (0.2 if self.soft_drop else 1.0)
        if self.fall_time >= speed:
            self.fall_time = 0
            m = Piece(self.current.x, self.current.y + 1, self.current.kind)
            m.rot = self.current.rot
            if self.board.valid(m):
                self.current = m
            else:
                self.lock_piece()


class Renderer:
    """Handles all drawing operations for the game."""

    def __init__(self, screen, font, font_big):
        """
        Initializes the Renderer.

        Args:
            screen: The main pygame.Surface to draw on.
            font (pygame.font.Font): The standard font for text.
            font_big (pygame.font.Font): The large font for titles.
        """
        self.screen = screen
        self.font = font
        self.font_big = font_big

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def draw_cell(self, x, y, color, offset_x, offset_y):
        """
        Draws a single block cell at a grid position with an offset.

        Args:
            x (int): Grid x-coordinate.
            y (int): Grid y-coordinate.
            color (tuple): RGB color.
            offset_x (int): Pixel offset for the x-axis.
            offset_y (int): Pixel offset for the y-axis.
        """
        px, py = offset_x + x * BLOCK, offset_y + y * BLOCK
        rect = pygame.Rect(px, py, BLOCK, BLOCK)
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, DARK, rect, 2)
    #pylint: disable= R0914
    def draw_window(self, game, offset_x, offset_y):
        """
        Draws the main game window, board, and side panel for one player.

        Args:
            game (Game): The game state to draw.
            offset_x (int): Pixel offset for the x-axis.
            offset_y (int): Pixel offset for the y-axis.
        """
        play_rect = (offset_x, offset_y, PLAY_W, PLAY_H)
        pygame.draw.rect(self.screen, (25, 25, 35), play_rect, border_radius=6)

        grid = game.board.create_grid()
        for y, row in enumerate(grid):
            for x, color in enumerate(row):
                if color:
                    self.draw_cell(x, y, color, offset_x, offset_y)
        for c in range(COLS + 1):
            x = offset_x + c * BLOCK
            pygame.draw.line(self.screen, GRID, (x, offset_y), (x, offset_y + PLAY_H))
        for r in range(ROWS + 1):
            y = offset_y + r * BLOCK
            pygame.draw.line(self.screen, GRID, (offset_x, y), (offset_x + PLAY_W, y))

        panel_x = offset_x + PLAY_W + MARGIN
        panel_rect = (panel_x, offset_y, SIDE, PLAY_H)
        pygame.draw.rect(self.screen, (20, 20, 28), panel_rect, border_radius=6)
        title = self.font_big.render("TETRIS", True, WHITE)
        self.screen.blit(title, (panel_x + 20, offset_y + 10))
        score_text = self.font.render(f"Score: {game.score}", True, WHITE)
        self.screen.blit(score_text, (panel_x + 20, offset_y + 80))

        self.draw_next(game.queue, panel_x, offset_y + 200)

    def draw_piece(self, piece, offset_x, offset_y):
        """Draws the currently falling piece."""
        for x, y in piece.cells():
            if y >= 0:
                self.draw_cell(x, y, piece.color, offset_x, offset_y)

    def draw_next(self, queue, px, py):
        """Draws the 'Next' piece queue in the side panel."""
        self.screen.blit(self.font.render("Next:", True, WHITE), (px, py))
        for idx, k in enumerate(queue[:3]):
            shape = SHAPES[k][0]
            color = COLORS[ORDER.index(k)]
            for r, row in enumerate(shape):
                for c, cell in enumerate(row):
                    if cell == '#':
                        sx = px + c * (BLOCK // 2)
                        sy = py + r * (BLOCK // 2) + idx * 70 + 30  # Offset for text
                        rect = pygame.Rect(sx, sy, BLOCK // 2, BLOCK // 2)
                        pygame.draw.rect(self.screen, color, rect)
                        pygame.draw.rect(self.screen, DARK, rect, 2)

    def draw_center_text(self, text, sub=None, offset_x=0, offset_y=0):
        """Draws text centered in the playfield (e.g., "GAME OVER")."""
        label = self.font_big.render(text, True, WHITE)
        self.screen.blit(label, (
            offset_x + PLAY_W // 2 - label.get_width() // 2,
            offset_y + PLAY_H // 2 - label.get_height() // 2
        ))
        if sub is not None:
            sub_label = self.font.render(sub, True, LIGHT)
            self.screen.blit(sub_label, (
                offset_x + PLAY_W // 2 - sub_label.get_width() // 2,
                offset_y + PLAY_H // 2 + 30
            ))


# for 2 players

# pylint: disable=too-many-instance-attributes
class AppDual:
    """
    Main application class for the 2-player (Human vs. Human,
    Human vs. Bot, or Bot vs. Bot) game.
    """

    def __init__(self, bot_enabled=False, bot=None, bot1=None, bot2=None):
        """
        Initializes the dual-player application.

        Args:
            bot_enabled (bool): True for Player vs. Bot mode.
            bot (Bot): The bot for Player vs. Bot mode (controls player 2).
            bot1 (Bot): The bot for player 1 (Bot vs. Bot mode).
            bot2 (Bot): The bot for player 2 (Bot vs. Bot mode).
        """
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH * 2, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 22)
        self.font_big = pygame.font.SysFont("consolas", 42, bold=True)

        self.games = [Game(), Game()]
        self.bot_enabled = bot_enabled  # pvb
        self.bot = bot  # 1b
        self.bot1 = bot1  # 1b for bvb
        self.bot2 = bot2  # 2b for bvb
        self.renderer = Renderer(self.screen, self.font, self.font_big)

        self.bot_move_interval = 0.3
        self.bot_timers = [0.0, 0.0]
        self.winner = None
        self.controls = [
            {'left': pygame.K_a, 'right': pygame.K_d, 'down': pygame.K_s,
             'rotate_cw': pygame.K_w, 'rotate_ccw': pygame.K_q, 'hard': pygame.K_e},
            {'left': pygame.K_LEFT, 'right': pygame.K_RIGHT, 'down': pygame.K_DOWN,
             'rotate_cw': pygame.K_UP, 'rotate_ccw': pygame.K_SLASH, 'hard': pygame.K_RSHIFT}
        ]

    def restart(self):
        """Resets both games to their initial state."""
        self.games = [Game(), Game()]
        self.bot_timers = [0.0, 0.0]
        self.winner = None

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def move_piece(self, game, dx=0, dy=0, rot=0, hard=False):
        """
        Applies a movement to a game's current piece, if valid.

        Args:
            game (Game): The game instance to modify.
            dx (int): Change in x.
            dy (int): Change in y.
            rot (int): Change in rotation.
            hard (bool): If True, perform a hard drop.
        """
        if hard:
            game.hard_drop()
            return
        piece = Piece(game.current.x + dx, game.current.y + dy, game.current.kind)
        piece.rot = (game.current.rot + rot) % len(game.current.shape)
        if game.board.valid(piece):
            game.current = piece
    # functions
    # _handle_player_input, _handle_events, _update_bots, _draw_frame
    # were rewrite from App,
    # but for 2 players with GPT
    def _handle_player_input(self, event):
        """Handles key presses for player movement."""
        for i, ctrl in enumerate(self.controls):
            game = self.games[i]

            is_bot = False
            if self.bot1 and self.bot2:  # bvb
                is_bot = True
            if i == 1 and self.bot_enabled and self.bot:  # pvb
                is_bot = True

            if is_bot or game.paused or game.over:
                continue

            if event.key == ctrl['left']:
                self.move_piece(game, dx=-1)
            elif event.key == ctrl['right']:
                self.move_piece(game, dx=1)
            elif event.key == ctrl['down']:
                game.soft_drop = True
            elif event.key == ctrl['rotate_cw']:
                self.move_piece(game, rot=+1)
            elif event.key == ctrl['rotate_ccw']:
                self.move_piece(game, rot=-1)
            elif event.key == ctrl['hard']:
                self.move_piece(game, hard=True)

    def _handle_events(self):
        """Handles all Pygame events (quit, restart, pause, input)."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                if event.key == pygame.K_r:
                    self.restart()
                elif event.key == pygame.K_m:
                    return False
                elif event.key == pygame.K_p:
                    for g in self.games:
                        g.paused = not g.paused
                else:
                    self._handle_player_input(event)

            elif event.type == pygame.KEYUP:
                for i, ctrl in enumerate(self.controls):
                    if event.key == ctrl['down']:
                        self.games[i].soft_drop = False
        return True

    def _update_bots(self, dt):
        """Updates bot timers and makes bot moves if ready."""
        for i in range(2):
            self.bot_timers[i] += dt

        if self.bot1 and self.bot2:
            # bvb
            for i, bot in enumerate([self.bot1, self.bot2]):
                game = self.games[i]
                if not game.over and not game.paused and self.bot_timers[i] >= self.bot_move_interval:
                    self.bot_timers[i] = 0.0
                    bot.best_move(game)
        elif self.bot_enabled and self.bot:
            # pvb
            game = self.games[1]
            if not game.over and not game.paused and self.bot_timers[1] >= self.bot_move_interval:
                self.bot_timers[1] = 0.0
                self.bot.best_move(game)

    def _update_winner(self):
        """Checks for a winner and updates the winner status."""
        if self.winner is None:
            if self.games[0].over and not self.games[1].over:
                self.winner = "Right Player Wins!"
            elif self.games[1].over and not self.games[0].over:
                self.winner = "Left Player Wins!"
            elif self.games[0].over and self.games[1].over:
                self.winner = "Draw!"

    def _draw_frame(self):
        """Draws the entire game frame (both players)."""
        self.screen.fill(BG)
        for i, game in enumerate(self.games):
            offset_x = MARGIN + i * WIDTH
            self.renderer.draw_window(game, offset_x, MARGIN)
            if not game.over and not game.paused:
                self.renderer.draw_piece(game.current, offset_x, MARGIN)

        if self.winner:
            for i in range(2):
                offset_x = MARGIN + i * WIDTH
                self.renderer.draw_center_text("GAME OVER", self.winner, offset_x, MARGIN)
                self.renderer.draw_center_text("R to restart", "M to menu", offset_x, MARGIN + 100)

        pygame.display.flip()

    def run(self):
        """The main game loop for the dual-player app."""
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0

            running = self._handle_events()
            if not running:
                break

            self._update_bots(dt)
            for g in self.games:
                g.step(dt)

            self._update_winner()

            self._draw_frame()

        pygame.quit()


# 1 player

# pylint: disable=too-many-instance-attributes
class App:
    """Main application class for a single-player game."""

    def __init__(self):
        """Initializes the single-player application."""
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 22)
        self.font_big = pygame.font.SysFont("consolas", 42, bold=True)
        self.game = Game()
        self.renderer = Renderer(self.screen, self.font, self.font_big)
        self.fullscreen = False
        self.bot_move_interval = 0.3
        self.bot_timer = 0.0

    def restart(self):
        """Resets the game to its initial state."""
        self.game = Game()
        self.bot_timer = 0.0

    def toggle_fullscreen(self):
        """Toggles the display between fullscreen and windowed mode."""
        self.fullscreen = not self.fullscreen
        info = pygame.display.Info()
        if self.fullscreen:
            res = (info.current_w, info.current_h)
            self.screen = pygame.display.set_mode(res, pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.renderer = Renderer(self.screen, self.font, self.font_big)

    def handle_player_input(self, key):
        """Handles key presses for player movement."""
        if self.game.paused or self.game.over:
            return

        if key == pygame.K_LEFT:
            m = Piece(self.game.current.x - 1, self.game.current.y, self.game.current.kind)
            m.rot = self.game.current.rot
            if self.game.board.valid(m):
                self.game.current = m
        elif key == pygame.K_RIGHT:
            m = Piece(self.game.current.x + 1, self.game.current.y, self.game.current.kind)
            m.rot = self.game.current.rot
            if self.game.board.valid(m):
                self.game.current = m
        elif key == pygame.K_DOWN:
            self.game.soft_drop = True
        elif key == pygame.K_SPACE:
            self.game.hard_drop()
        elif key in (pygame.K_UP, pygame.K_x):
            r = self.game.current.rotated(+1)
            if self.game.board.valid(r):
                self.game.current = r
        elif key == pygame.K_z:
            r = self.game.current.rotated(-1)
            if self.game.board.valid(r):
                self.game.current = r

    def handle_events(self, bot_enabled):
        """Handles all Pygame events (quit, restart, pause, input)."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return False
                if event.key == pygame.K_m:
                    return False
                if event.key == pygame.K_r:
                    self.restart()
                    if not bot_enabled:
                        self.game.fall_interval = BASE_FALL_TIME
                elif event.key == pygame.K_p:
                    self.game.paused = not self.game.paused
                elif event.key == pygame.K_f:
                    self.toggle_fullscreen()
                elif not bot_enabled:
                    self.handle_player_input(event.key)
            elif event.type == pygame.KEYUP and not bot_enabled:
                if event.key == pygame.K_DOWN:
                    self.game.soft_drop = False
        return True

    def update_bot(self, dt, bot):
        """Updates the bot's timer and makes a move if ready."""
        if bot and not self.game.over and not self.game.paused:
            self.bot_timer += dt
            if self.bot_timer >= self.bot_move_interval:
                self.bot_timer = 0.0
                bot.best_move(self.game)

    def draw_frame(self):
        """Draws the entire game frame."""
        self.screen.fill(BG)
        self.renderer.draw_window(self.game, MARGIN, MARGIN)

        if not self.game.over and not self.game.paused:
            self.renderer.draw_piece(self.game.current, MARGIN, MARGIN)
        elif self.game.paused:
            self.renderer.draw_center_text("P to resume", offset_x=MARGIN, offset_y=MARGIN)
        elif self.game.over:
            self.renderer.draw_center_text("R to restart/M to menu", offset_x=MARGIN, offset_y=MARGIN)

        pygame.display.flip()

    def run(self, bot_enabled=False, bot=None):
        """
        The main game loop for the single-player app.

        Args:
            bot_enabled (bool): True if a bot is playing.
            bot (Bot): The bot instance to use if bot_enabled is True.
        """
        running = True
        if bot_enabled and bot:
            self.bot_move_interval = 0.05
            self.game.fall_interval = 0.1
        else:
            self.bot_move_interval = 0.3
            self.game.fall_interval = BASE_FALL_TIME

        while running:
            dt = self.clock.tick(FPS) / 1000.0
            running = self.handle_events(bot_enabled)
            if not running:
                break

            if bot_enabled:
                self.update_bot(dt, bot)
            self.game.step(dt)
            self.draw_frame()

        pygame.quit()

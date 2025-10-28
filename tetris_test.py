import pygame, random

COLS, ROWS = 10, 20
BLOCK = 32
PLAY_W, PLAY_H = COLS * BLOCK, ROWS * BLOCK
SIDE, MARGIN = 200, 20
WIDTH, HEIGHT = PLAY_W + SIDE + MARGIN * 3, PLAY_H + MARGIN * 2
FPS = 60
BASE_FALL_TIME = 0.75

WHITE = (240,240,240)
LIGHT = (210,210,210)
DARK = (30,30,30)
BG = (15,15,20)
GRID = (45,45,55)

COLORS = [
    (80,190,230),(240,240,90),(240,150,80),(100,180,90),
    (200,100,200),(70,90,200),(240,90,100)
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

ORDER = ['I','O','L','S','T','J','Z']

class Piece:
    def __init__(self, x, y, kind):
        self.x = x
        self.y = y
        self.kind = kind
        self.rot = 0
        self.shape = SHAPES[kind]
        self.color = COLORS[ORDER.index(kind)]

    def cells(self):
        pattern = self.shape[self.rot]
        positions = []
        x_offset, y_offset = 1, 1
        if self.kind == 'I' and self.rot == 1: x_offset = 2
        for r in range(4):
            for c in range(4):
                if pattern[r][c] == '#':
                    positions.append((int(self.x + c - x_offset), int(self.y + r - y_offset)))
        return positions

    def rotated(self, dir=1):
        p = Piece(self.x, self.y, self.kind)
        p.rot = (self.rot + dir) % len(self.shape)
        return p

class Board:
    def __init__(self):
        self.locked = {}
        self.cleared_lines = 0

    def create_grid(self):
        grid = [[None for _ in range(COLS)] for _ in range(ROWS)]
        for (x,y), color in self.locked.items():
            if 0<=x<COLS and 0<=y<ROWS:
                grid[y][x] = color
        return grid

    def valid(self, piece):
        for x, y in piece.cells():
            if x<0 or x>=COLS or y>=ROWS: return False
            if y>=0 and (x,y) in self.locked: return False
        return True

    def add_piece(self, piece):
        for x, y in piece.cells():
            self.locked[(x,y)] = piece.color

    def check_lost(self):
        return any(y<0 for _,y in self.locked)

    def clear_rows(self):
        full_rows = [y for y in range(ROWS) if all((x,y) in self.locked for x in range(COLS))]
        if not full_rows: return 0
        for y in full_rows:
            for x in range(COLS):
                self.locked.pop((x,y))
        new_locked = {}
        for (x,y), color in self.locked.items():
            shift = sum(1 for ry in full_rows if ry>y)
            new_locked[(x,y+shift)] = color
        self.locked = new_locked
        self.cleared_lines += len(full_rows)
        return len(full_rows)

class Game:
    def __init__(self):
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
        bag = ORDER.copy()
        random.shuffle(bag)
        return bag

    def get_next(self):
        while len(self.queue)<3: self.queue.extend(self.new_bag())
        return Piece(COLS//2,-1,self.queue.pop(0))

    def peek_next(self,n=3):
        while len(self.queue)<n: self.queue.extend(self.new_bag())
        return self.queue[:n]

    def hard_drop(self):
        while True:
            m = Piece(self.current.x, self.current.y+1, self.current.kind)
            m.rot = self.current.rot
            if self.board.valid(m):
                self.current = m
            else: break
        self.lock_piece()

    def lock_piece(self):
        self.board.add_piece(self.current)
        cleared = self.board.clear_rows()
        if cleared:
            self.lines += cleared
            self.score += {0:0,1:100,2:300,3:500,4:800}[cleared]
        self.current = self.get_next()
        self.next_piece = self.peek_next()
        if not self.board.valid(self.current) or self.board.check_lost(): self.over = True

    def step(self, dt):
        if self.paused or self.over: return
        self.fall_time += dt
        speed = self.fall_interval*(0.2 if self.soft_drop else 1.0)
        if self.fall_time>=speed:
            self.fall_time=0
            m=Piece(self.current.x, self.current.y+1,self.current.kind)
            m.rot=self.current.rot
            if self.board.valid(m):
                self.current=m
            else: self.lock_piece()


class Renderer:
    def __init__(self, screen, font, font_big):
        self.screen = screen
        self.font = font
        self.font_big = font_big

    def draw_cell(self, x, y, color, offset_x, offset_y):
        px, py = offset_x + x*BLOCK, offset_y + y*BLOCK
        rect = pygame.Rect(px, py, BLOCK, BLOCK)
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, DARK, rect, 2)

    def draw_window(self, game, offset_x, offset_y):

        pygame.draw.rect(self.screen,(25,25,35),(offset_x,offset_y,PLAY_W,PLAY_H),border_radius=6)
        grid = game.board.create_grid()
        for y in range(ROWS):
            for x in range(COLS):
                if grid[y][x]: self.draw_cell(x,y,grid[y][x],offset_x,offset_y)
        for c in range(COLS+1):
            x = offset_x + c*BLOCK
            pygame.draw.line(self.screen, GRID, (x,offset_y),(x,offset_y+PLAY_H))
        for r in range(ROWS+1):
            y = offset_y + r*BLOCK
            pygame.draw.line(self.screen, GRID, (offset_x,y),(offset_x+PLAY_W,y))

        panel_x = offset_x + PLAY_W + MARGIN
        pygame.draw.rect(self.screen,(20,20,28),(panel_x,offset_y,SIDE,PLAY_H),border_radius=6)
        self.screen.blit(self.font_big.render("TETRIS",True,WHITE),(panel_x+20,offset_y+10))
        self.screen.blit(self.font.render(f"Score: {game.score}",True,WHITE),(panel_x+20,offset_y+80))

        self.draw_next(game.queue, panel_x, offset_y+200)

    def draw_piece(self, piece, offset_x, offset_y):
        for x,y in piece.cells():
            if y>=0: self.draw_cell(x,y,piece.color, offset_x, offset_y)

    def draw_next(self, queue, px, py):
        self.screen.blit(self.font.render("Next:",True,WHITE),(px,py))
        for idx,k in enumerate(queue[:3]):
            shape,color=SHAPES[k][0],COLORS[ORDER.index(k)]
            for r in range(4):
                for c in range(4):
                    if shape[r][c]=='#':
                        sx,sy = px+c*(BLOCK//2), py+r*(BLOCK//2)+idx*70
                        rect = pygame.Rect(sx,sy,BLOCK//2,BLOCK//2)
                        pygame.draw.rect(self.screen,color,rect)
                        pygame.draw.rect(self.screen,DARK,rect,2)

    def draw_center_text(self, text, sub=None, offset_x=0, offset_y=0):
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



class AppDual:
    def __init__(self, bot_enabled=False, bot=None, bot1=None, bot2=None):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH * 2, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 22)
        self.font_big = pygame.font.SysFont("consolas", 42, bold=True)

        self.games = [Game(), Game()]
        self.bot_enabled = bot_enabled   # pvb
        self.bot = bot                   # 1b
        self.bot1 = bot1                 # 1b for bvb
        self.bot2 = bot2                 # 2b for bvb
        self.renderer = Renderer(self.screen, self.font, self.font_big)

        self.controls = [
            {'left': pygame.K_a, 'right': pygame.K_d, 'down': pygame.K_s,
             'rotate_cw': pygame.K_w, 'rotate_ccw': pygame.K_q, 'hard': pygame.K_e},
            {'left': pygame.K_LEFT, 'right': pygame.K_RIGHT, 'down': pygame.K_DOWN,
             'rotate_cw': pygame.K_UP, 'rotate_ccw': pygame.K_SLASH, 'hard': pygame.K_RSHIFT}
        ]

    def restart(self):
        self.games = [Game(), Game()]

    def move_piece(self, game, dx=0, dy=0, rot=0, hard=False):
        if hard:
            game.hard_drop()
            return
        piece = Piece(game.current.x + dx, game.current.y + dy, game.current.kind)
        piece.rot = (game.current.rot + rot) % len(game.current.shape)
        if game.board.valid(piece):
            game.current = piece

    def run(self):
        running = True
        winner = None

        while running:
            dt = self.clock.tick(FPS) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_r:
                        self.restart()
                        winner = None
                    elif event.key == pygame.K_m:
                        return
                    elif event.key == pygame.K_p:
                        for g in self.games:
                            g.paused = not g.paused

                    for i, ctrl in enumerate(self.controls):
                        game = self.games[i]
                        if game.paused or game.over:
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

                elif event.type == pygame.KEYUP:
                    for i, ctrl in enumerate(self.controls):
                        if event.key == ctrl['down']:
                            self.games[i].soft_drop = False

            if self.bot1 and self.bot2:
                # bvb
                for i, bot in enumerate([self.bot1, self.bot2]):
                    game = self.games[i]
                    if not game.over and not game.paused:
                        best_piece, _ = bot.best_move(game)
                        if best_piece:
                            game.current = best_piece

            elif self.bot_enabled and self.bot:
                # pvb
                game = self.games[1]
                if not game.over and not game.paused:
                    best_piece, _ = self.bot.best_move(game)
                    if best_piece:
                        game.current = best_piece

            for g in self.games:
                g.step(dt)

            if winner is None:
                if self.games[0].over and not self.games[1].over:
                    winner = "Right Player Wins!"
                elif self.games[1].over and not self.games[0].over:
                    winner = "Left Player Wins!"
                elif self.games[0].over and self.games[1].over:
                    winner = "Draw!"

            self.screen.fill(BG)
            for i, game in enumerate(self.games):
                offset_x = i * (PLAY_W + SIDE + MARGIN * 2)
                self.renderer.draw_window(game, offset_x, MARGIN)
                if not game.over and not game.paused:
                    self.renderer.draw_piece(game.current, offset_x, MARGIN)

            if winner:
                for i in range(2):
                    offset_x = i * (PLAY_W + SIDE + MARGIN * 2)
                    self.renderer.draw_center_text("GAME OVER", winner, offset_x, MARGIN)
                    self.renderer.draw_center_text("Press R to restart", "Press M to menu", offset_x, MARGIN + 100)

            pygame.display.flip()

        pygame.quit()


class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH,HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas",22)
        self.font_big = pygame.font.SysFont("consolas",42,bold=True)
        self.game = Game()
        self.renderer = Renderer(self.screen,self.font,self.font_big)
        self.fullscreen = False

    def restart(self):
        self.game = Game()

    def run(self, bot_enabled=False, bot=None):
        running=True
        while running:
            dt = self.clock.tick(FPS)/1000.0
            for event in pygame.event.get():
                if event.type==pygame.QUIT: running=False
                elif event.type==pygame.KEYDOWN:
                    if event.key in (pygame.K_q,pygame.K_ESCAPE): running=False
                    elif event.key==pygame.K_r: self.restart()
                    elif event.key==pygame.K_p: self.game.paused=not self.game.paused
                    elif event.key==pygame.K_f:
                        self.fullscreen=not self.fullscreen
                        info=pygame.display.Info()
                        if self.fullscreen:
                            self.screen=pygame.display.set_mode((info.current_w,info.current_h),pygame.FULLSCREEN)
                        else:
                            self.screen=pygame.display.set_mode((WIDTH,HEIGHT))
                        self.renderer = Renderer(self.screen,self.font,self.font_big)
                    elif not self.game.paused and not self.game.over:
                        if event.key==pygame.K_LEFT:
                            m=Piece(self.game.current.x-1,self.game.current.y,self.game.current.kind);m.rot=self.game.current.rot
                            if self.game.board.valid(m): self.game.current=m
                        elif event.key==pygame.K_RIGHT:
                            m=Piece(self.game.current.x+1,self.game.current.y,self.game.current.kind);m.rot=self.game.current.rot
                            if self.game.board.valid(m): self.game.current=m
                        elif event.key==pygame.K_DOWN: self.game.soft_drop=True
                        elif event.key==pygame.K_SPACE: self.game.hard_drop()
                        elif event.key in (pygame.K_UP,pygame.K_x):
                            r=self.game.current.rotated(+1)
                            if self.game.board.valid(r): self.game.current=r
                        elif event.key==pygame.K_z:
                            r=self.game.current.rotated(-1)
                            if self.game.board.valid(r): self.game.current=r
                elif event.type==pygame.KEYUP and event.key==pygame.K_DOWN:
                    self.game.soft_drop=False

            if bot_enabled and bot:
                best_piece, metrics = bot.best_move()
                if best_piece:
                    self.game.current = best_piece


            self.game.step(dt)
            self.renderer.draw_window(self.game)
            if not self.game.over and not self.game.paused:
                self.renderer.draw_piece(self.game.current)
                self.renderer.draw_next(self.game.queue)
            elif self.game.paused:
                self.renderer.draw_center_text("P to resume")
            elif self.game.over:
                self.renderer.draw_center_text("R to restart/Esc to quit")

            pygame.display.flip()

        pygame.quit()

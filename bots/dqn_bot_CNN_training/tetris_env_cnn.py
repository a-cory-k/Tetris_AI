import gymnasium as gym
from gymnasium import spaces
import numpy as np
from tetris_test import Game, ROWS, COLS, ORDER
import copy


class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self):
        super().__init__()
        self.game = Game()
        self.action_space = spaces.Discrete(4 * COLS)  # 40 
        self.observation_space = spaces.Dict({
            'board': spaces.Box(low=0.0, high=1.0, shape=(1, ROWS, COLS), dtype=np.float32),
            'kind': spaces.Discrete(len(ORDER))
        })
        self.height_penalty_factor = 0.02
        self.hole_penalty_factor = 0.01

    def _calculate_max_height(self, game_obj):
        grid = game_obj.board.create_grid()
        max_height = 0
        for c in range(COLS):
            col_height = 0
            for r in range(ROWS):
                if grid[r][c] is not None:
                    col_height = ROWS - r
                    break
            if col_height > max_height:
                max_height = col_height
        return max_height

    def _calculate_holes(self, game_obj):
        grid = game_obj.board.create_grid()
        holes = 0
        for c in range(COLS):
            block_above = False
            for r in range(ROWS):
                if grid[r][c] is not None:
                    block_above = True
                elif block_above:
                    holes += 1
        return holes
    def reset(self, seed=None, options=None):
        self.game = self.game.__class__()
        return self._get_state(), {}

    def _get_state_from_game(self, game_obj):
        grid = game_obj.board.create_grid()
        binary_grid = np.zeros(shape=(ROWS, COLS), dtype=np.float32)
        for r in range(ROWS):
            for c in range(COLS):
                if grid[r][c] is not None:
                    binary_grid[r][c] = 1.0
        board_state = np.expand_dims(binary_grid, axis=0)
        try:
            current_kind_str = game_obj.current.kind
            kind_idx = ORDER.index(current_kind_str)
        except (AttributeError, ValueError, IndexError):
            kind_idx = 0
        return {'board': board_state, 'kind': kind_idx}

    def _get_state(self):
        return self._get_state_from_game(self.game)

    def _calculate_reward(self, game_obj, lines_cleared, done):
        reward = 1.0
        if lines_cleared == 1:
            reward += 40
        elif lines_cleared == 2:
            reward += 100
        elif lines_cleared == 3:
            reward += 300
        elif lines_cleared == 4:
            reward += 1200

        max_height = self._calculate_max_height(game_obj)
        height_penalty = self.height_penalty_factor * (max_height ** 2)
        reward -= height_penalty

        num_holes = self._calculate_holes(game_obj)
        hole_penalty = self.hole_penalty_factor * num_holes
        reward -= hole_penalty
        if done:
            reward -= 5.0

        return reward

    def step(self, action):
        rot, col = divmod(action, COLS)
        piece = self.game.current.rotated(rot)
        piece.x = col
        lines_before = self.game.lines

        valid_move = self.game.board.valid(piece)
        if valid_move:
            self.game.current = piece
        self.game.hard_drop()

        lines_cleared = self.game.lines - lines_before
        done = self.game.over

        reward = self._calculate_reward(self.game, lines_cleared, done)

        return self._get_state(), reward, done, False, {}

    def get_next_states(self):
        next_states_info = {}
        current_kind_idx = ORDER.index(self.game.current.kind) if self.game.current else 0

        for action in range(self.action_space.n):
            sim_game = copy.deepcopy(self.game)
            rot, col = divmod(action, COLS)
            piece = sim_game.current.rotated(rot)
            piece.x = col
            lines_before = sim_game.lines

            valid_move = sim_game.board.valid(piece)
            if valid_move:
                sim_game.current = piece
            sim_game.hard_drop()

            lines_cleared = sim_game.lines - lines_before
            done = sim_game.over

            reward = self._calculate_reward(sim_game, lines_cleared, done)
            next_state_dict = self._get_state_from_game(sim_game)
            next_states_info[action] = (next_state_dict, reward, done)

        return next_states_info, current_kind_idx


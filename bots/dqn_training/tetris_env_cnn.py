"""
A custom Gymnasium (formerly Gym) environment for Tetris.
"""
import copy
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from app.tetris_dual import Game, ROWS, COLS, ORDER

class TetrisEnv(gym.Env):
    """
    This environment implements the standard gym.Env interface for a simplified
    version of Tetris. The agent chooses a final (rotation, column) position
    for the current piece, and the piece is then hard-dropped.

    Attributes:
        action_space (gym.spaces.Discrete): The action space, representing
            4 rotations * COLS possible column placements (e.g., 4 * 10 = 40).
        observation_space (gym.spaces.Dict): The observation space, containing:
            - 'board': A (1, ROWS, COLS) binary numpy array representing the
              game board (1.0 for occupied, 0.0 for empty).
            - 'kind': A (gym.spaces.Discrete) integer representing the index
              of the current falling tetromino (based on the ORDER list).
        game (Game): An instance of the underlying Tetris game logic
            (from tools.tetris_for_gym).
        height_penalty_factor (float): Multiplier for the height penalty.
        hole_penalty_factor (float): Multiplier for the hole penalty.
    """
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self):
        """Initializes the Tetris environment.

        Sets up the game instance, defines the action and observation spaces,
        and initializes reward shaping parameters (penalty factors for height
        and holes).
        """
        super().__init__()
        self.game = Game()
        self.action_space = spaces.Discrete(4 * COLS)  # 40
        self.observation_space = spaces.Dict({
            'board': spaces.Box(low=0.0, high=1.0, shape=(1, ROWS, COLS), dtype=np.float32),
            'kind': spaces.Discrete(len(ORDER))
        })
        self.height_penalty_factor = 0.02
        self.hole_penalty_factor = 0.01

    def calculate_max_height(self, game_obj):
        """Calculates the maximum height of the pieces on the board.

        The height is defined as the number of rows from the bottom to the
        highest occupied cell across all columns.

        Args:
            game_obj (Game): The game instance to evaluate.

        Returns:
            int: The maximum height (stack height) on the board.
        """
        grid = game_obj.board.create_grid()
        max_height = 0
        for c in range(COLS):
            col_height = 0
            for r in range(ROWS):
                if grid[r][c] is not None:
                    col_height = ROWS - r
                    break
            max_height = max(max_height, col_height)
        return max_height

    def calculate_holes(self, game_obj):
        """Calculates the total number of holes on the board.

        A hole is defined as an empty cell (None) that has at least one
        occupied cell (not None) above it in the same column.

        Args:
            game_obj (Game): The game instance to evaluate.

        Returns:
            int: The total number of holes.
        """
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

    def reset(self, *, seed=None, options=None):
        """Resets the environment to the initial state (starts a new game).

        Conforms to the standard gym.Env.reset() interface.

        Args:
            seed (Optional[int]): The seed for the random number generator.
                (Note: The underlying game logic might not use this seed).
            options (Optional[dict]): Additional options (not used here).

        Returns:
            tuple: A tuple containing:
                - dict: The initial observation (`get_state()`).
                - dict: An empty info dictionary.
        """
        self.game = self.game.__class__()
        return self.get_state(), {}

    def get_state_from_game(self, game_obj):
        """Converts a game instance into the standard observation format.

        Args:
            game_obj (Game): The game instance to extract the state from.

        Returns:
            dict: An observation dictionary matching `self.observation_space`:
                - 'board': (1, ROWS, COLS) np.float32 binary grid.
                - 'kind': int index of the current tetromino.
        """
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

    def get_state(self):
        """Gets the observation for the current game state.

        Returns:
            dict: The observation dictionary from `get_state_from_game`
            applied to the current `self.game` instance.
        """
        return self.get_state_from_game(self.game)

    def calculate_reward(self, game_obj, lines_cleared, done):
        """Calculates the reward based on the game state after an action.

        The reward includes:
        - A large bonus for clearing lines (scales with lines cleared).
        - A small positive reward (1.0) for surviving the step.
        - A quadratic penalty for the maximum height of the stack.
        - A linear penalty for the number of holes.
        - A penalty for losing the game (game over).

        Args:
            game_obj (Game): The game instance after the action.
            lines_cleared (int): Number of lines cleared in this step.
            done (bool): Whether the game ended in this step.

        Returns:
            float: The calculated reward.
        """
        reward = 1.0
        if lines_cleared == 1:
            reward += 40
        elif lines_cleared == 2:
            reward += 100
        elif lines_cleared == 3:
            reward += 300
        elif lines_cleared == 4:
            reward += 1200

        max_height = self.calculate_max_height(game_obj)
        height_penalty = self.height_penalty_factor * (max_height ** 2)
        reward -= height_penalty

        num_holes = self.calculate_holes(game_obj)
        hole_penalty = self.hole_penalty_factor * num_holes
        reward -= hole_penalty
        if done:
            reward -= 5.0

        return reward

    def render(self):
        """Optional."""
    def step(self, action):
        """Executes one time step in the environment.

        The agent provides an action (0-39), which is decoded into a
        rotation (0-3) and a column (0-9). The piece is moved to that
        position (if valid) and hard-dropped.

        Conforms to the standard gym.Env.step() interface.

        Args:
            action (int): The action chosen by the agent (0 to 4 * COLS - 1).

        Returns:
            tuple: A tuple containing:
                - dict: The next observation.
                - float: The reward obtained.
                - bool: `terminated` (True if the game is over, else False).
                - bool: `truncated` (Always False, as truncation is not implemented).
                - dict: An empty info dictionary.
        """
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
        reward = self.calculate_reward(self.game, lines_cleared, done)
        return self.get_state(), reward, done, False, {}

    def get_next_states(self):
        """Simulates all possible next states for the current piece.

        This method iterates through all actions in the action space
        (e.g., 40 moves), simulates the hard drop for each, and calculates
        the resulting state, reward, and 'done' status.

        This is useful for lookahead algorithms (e.g., MCTS) or agents that
        evaluate all possible moves before selecting one.

        Returns:
            tuple: A tuple containing:
                - dict[int, tuple]: A dictionary mapping action (int) to a
                  tuple of (next_state_dict, reward, done).
                - int: The index (kind) of the current piece that was played.
        """
        next_states_info = {}
        try:
            current_kind_idx = ORDER.index(self.game.current.kind)
        except (AttributeError, ValueError, IndexError):
            current_kind_idx = 0
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
            reward = self.calculate_reward(sim_game, lines_cleared, done)
            next_state_dict = self.get_state_from_game(sim_game)
            next_states_info[action] = (next_state_dict, reward, done)

        return next_states_info, current_kind_idx

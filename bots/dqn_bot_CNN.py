"""
DQN (Deep Q-Network) Bot implementation for playing Tetris using a CNN.

This module defines the DQNBot class, which loads a pre-trained CNN model
to determine the best move in a game of Tetris.
"""

import sys
from pathlib import Path
import torch
import numpy as np
from bots.cnn_training.tetris_train_cnn import DeepQNetwork, GAMMA, DEVICE
from bots.cnn_training.tetris_env_cnn import TetrisEnv
from app.tetris_dual import  ROWS, COLS, ORDER

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# pylint: disable=too-few-public-methods
class DQNBot:
    """
    A bot that uses a Deep Q-Network (DQN) with a CNN to play Tetris.
    """
    def __init__(self, model_path: str):
        """
        Initializes the DQNBot and loads the pre-trained model.

        Args:
            model_path (str): The file path to the saved model's state dictionary.
        """
        self.device = DEVICE
        self.adapter_env = TetrisEnv()
        action_dim = self.adapter_env.action_space.n
        num_kinds = len(ORDER)
        self.agent_model = DeepQNetwork(
            h=ROWS,
            w=COLS,
            num_kinds=num_kinds,
            num_actions=action_dim
        ).to(self.device)

        try:
            self.agent_model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Warning: Could not load model from {model_path}. Error: {e}")
        self.agent_model.eval()

    # pylint: disable=too-many-locals
    def best_move(self, game):
        """
        Calculates and performs the best move using the DQN model.

        It queries the environment for all possible next states, evaluates
        them with the Q-network, and selects the action leading to the
        state with the highest Q-value.

        Args:
            game: The current game state object.

        Returns:
            A tuple (None, {}), matching the expected return format.
        """
        self.adapter_env.game = game
        try:
            next_states_info, _ = self.adapter_env.get_next_states()
        except Exception as e:  #pylint: disable=W0718
            print(f"Error getting next states: {e}")
            return None, {}

        if not next_states_info:
            game.hard_drop()
            return None, {}

        all_next_boards = np.concatenate(
            [info[0]['board'] for info in next_states_info.values()], axis=0
        )
        all_next_kinds = np.array(
            [info[0]['kind'] for info in next_states_info.values()], dtype=np.int64
        )
        all_rewards = np.array(
            [info[1] for info in next_states_info.values()], dtype=np.float32
        )
        all_dones = np.array(
            [info[2] for info in next_states_info.values()], dtype=np.float32
        )

        next_boards_v = torch.from_numpy(all_next_boards).float().to(self.device)
        next_kinds_v = torch.from_numpy(all_next_kinds).long().to(self.device)
        rewards_v = torch.from_numpy(all_rewards).float().to(self.device)
        dones_v = torch.from_numpy(all_dones).float().to(self.device)
        next_state_input = {'board': next_boards_v, 'kind': next_kinds_v}

        with torch.no_grad():
            next_q_values = self.agent_model(next_state_input)
            next_q_values_max = next_q_values.max(dim=1)[0]
            next_q_values_max = next_q_values_max * (1.0 - dones_v)
            total_q_values = rewards_v + (GAMMA * next_q_values_max)

        action = int(torch.argmax(total_q_values).item())
        rot, col = divmod(action, COLS)

        new_piece = game.current.rotated(rot)
        new_piece.x = col
        if game.board.valid(new_piece):
            game.current = new_piece
        game.hard_drop()

        return None, {}

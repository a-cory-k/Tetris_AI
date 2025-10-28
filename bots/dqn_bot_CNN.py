import torch
import numpy as np
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from tetris_train_cnn import DeepQNetwork, GAMMA, DEVICE
from tetris_env_cnn import TetrisEnv, ROWS, COLS, ORDER
from tetris_test import Piece

class DQNBot:
    def __init__(self, model_path: str):

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
        except Exception as e:
            print("no")

        self.agent_model.eval()

    def best_move(self, game):

        self.adapter_env.game = game
        try:
            next_states_info, _ = self.adapter_env.get_next_states()
        except Exception as e:
            return None, {} 

        all_next_boards = np.concatenate([info[0]['board'] for info in next_states_info.values()], axis=0)
        all_next_kinds = np.array([info[0]['kind'] for info in next_states_info.values()], dtype=np.int64)
        all_rewards = np.array([info[1] for info in next_states_info.values()], dtype=np.float32)
        all_dones = np.array([info[2] for info in next_states_info.values()], dtype=np.float32)

        next_boards_v = torch.from_numpy(all_next_boards).float().to(self.device)
        next_kinds_v = torch.from_numpy(all_next_kinds).long().to(self.device)
        rewards_v = torch.from_numpy(all_rewards).float().to(self.device)
        dones_v = torch.from_numpy(all_dones).float().to(self.device)
        next_state_input = {'board': next_boards_v, 'kind': next_kinds_v}

        with torch.no_grad():
            next_q_values_max = self.agent_model(next_state_input).max(dim=1)[0]
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

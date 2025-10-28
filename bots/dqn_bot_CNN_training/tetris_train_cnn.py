import os
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict

from tetris_env_cnn import TetrisEnv, ROWS, COLS, ORDER

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeepQNetwork(nn.Module):
    def __init__(self, h=ROWS, w=COLS, num_kinds=len(ORDER), num_actions=4 * COLS):
        super().__init__()
        input_channels = 1
        self.num_actions = num_actions
        self.num_kinds = num_kinds
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, h, w)
            conv_out_size = self.conv(dummy_input).shape[1]  # 640
        self.fc_shared = nn.Sequential(
            nn.Linear(conv_out_size + num_kinds, 256), nn.ReLU()
        )
        self.fc_output = nn.Linear(256, num_actions)

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        board_tensor = state_dict['board']
        kind_tensor = state_dict['kind']
        if board_tensor.dim() == 3: board_tensor = board_tensor.unsqueeze(1)
        if board_tensor.shape[1] != 1: board_tensor = board_tensor.reshape(-1, 1, ROWS, COLS)
        conv_features = self.conv(board_tensor)
        kind_onehot = F.one_hot(kind_tensor.long(), num_classes=self.num_kinds).float()
        if kind_onehot.dim() == 1: kind_onehot = kind_onehot.unsqueeze(0)
        combined_features = torch.cat([conv_features, kind_onehot], dim=1)
        shared_output = self.fc_shared(combined_features)
        q = self.fc_output(shared_output)
        return q

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state_dict, action, reward, next_state_dict, done):
        self.memory.append((state_dict, action, reward, next_state_dict, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        boards = np.concatenate([s['board'] for s in states], axis=0)
        kinds = np.array([s['kind'] for s in states], dtype=np.int64)
        next_boards = np.concatenate([s['board'] for s in next_states], axis=0)
        next_kinds = np.array([s['kind'] for s in next_states], dtype=np.int64)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        batch_dict = (
            {'board': boards, 'kind': kinds},
            actions,
            rewards,
            {'board': next_boards, 'kind': next_kinds},
            dones
        )
        return batch_dict

    def __len__(self):
        return len(self.memory)



# hyperparams
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 64
MEMORY_SIZE = 30000
TARGET_UPDATE = 500
EPS_START = 1
EPS_END = 0.05
EPISODES = 2000
EPS_DECAY_FRAMES = 10000.0

CHECKPOINT_FILE = "tetris_checkpoint_lookahead_dqn.pth"
BEST_MODEL_FILE = "dqn_tetris_best_lookahead_dqn.pth"

def epsilon_by_frame(frame_idx):
    return max(EPS_END, EPS_START - frame_idx * (EPS_START - EPS_END) / EPS_DECAY_FRAMES)


def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE: return None

    (states_dict, actions, rewards, next_states_dict, dones) = memory.sample(BATCH_SIZE)
    states_board_v = torch.from_numpy(states_dict['board']).float().to(DEVICE)
    states_kind_v = torch.from_numpy(states_dict['kind']).long().to(DEVICE)
    next_states_board_v = torch.from_numpy(next_states_dict['board']).float().to(DEVICE)
    next_states_kind_v = torch.from_numpy(next_states_dict['kind']).long().to(DEVICE)
    actions_v = torch.from_numpy(actions).long().to(DEVICE).unsqueeze(1)
    rewards_v = torch.from_numpy(rewards).float().to(DEVICE)
    dones_v = torch.from_numpy(dones).float().to(DEVICE)

    current_state_input = {'board': states_board_v, 'kind': states_kind_v}
    next_state_input = {'board': next_states_board_v, 'kind': next_states_kind_v}
    q_values = policy_net(current_state_input).gather(1, actions_v).squeeze(1)
    with torch.no_grad():
        next_q_values_max = target_net(next_state_input).max(dim=1)[0]
        next_q_values_max = next_q_values_max * (1.0 - dones_v)
        target_q_values = rewards_v + (GAMMA * next_q_values_max)

    loss_fn = nn.MSELoss()
    loss = loss_fn(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


if __name__ == '__main__':
    print("Device:", DEVICE)
    env = TetrisEnv()
    action_dim = env.action_space.n
    print(f"Action dimensions: {action_dim}")

    policy_net = DeepQNetwork(h=ROWS, w=COLS, num_kinds=len(ORDER), num_actions=action_dim).to(DEVICE)
    target_net = DeepQNetwork(h=ROWS, w=COLS, num_kinds=len(ORDER), num_actions=action_dim).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(MEMORY_SIZE)
    start_ep = 1
    frame_idx = 0
    best_reward = -float('inf')

    if os.path.exists(CHECKPOINT_FILE):
        print(f"Loading RL checkpoint from {CHECKPOINT_FILE}...")
        try:
            checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE, weights_only=False)
            policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            target_net.load_state_dict(checkpoint['target_net_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            memory = checkpoint['memory']
            start_ep = checkpoint['epoch'] + 1
            frame_idx = checkpoint['frame_idx']
            best_reward = checkpoint.get('best_reward', -float('inf'))
            print(f"*** Resuming RL training from episode {start_ep} (frame {frame_idx}) ***")
        except Exception as e:
            print(f"Error loading RL checkpoint, starting fresh: {e}.")
            memory = ReplayMemory(MEMORY_SIZE)

    for ep in range(start_ep, EPISODES + 1):
        state_dict, _ = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            eps = epsilon_by_frame(frame_idx)
            action = 0
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                next_states_info, current_kind_idx = env.get_next_states()

                all_next_boards = np.concatenate([info[0]['board'] for info in next_states_info.values()], axis=0)
                all_next_kinds = np.array([info[0]['kind'] for info in next_states_info.values()], dtype=np.int64)

                all_rewards = np.array([info[1] for info in next_states_info.values()], dtype=np.float32)
                all_dones = np.array([info[2] for info in next_states_info.values()], dtype=np.float32)

                next_boards_v = torch.from_numpy(all_next_boards).float().to(DEVICE)
                next_kinds_v = torch.from_numpy(all_next_kinds).long().to(DEVICE)
                rewards_v = torch.from_numpy(all_rewards).float().to(DEVICE)
                dones_v = torch.from_numpy(all_dones).float().to(DEVICE)

                next_state_input = {'board': next_boards_v, 'kind': next_kinds_v}
                with torch.no_grad():
                    next_q_values_max = policy_net(next_state_input).max(dim=1)[0]
                    next_q_values_max = next_q_values_max * (1.0 - dones_v)
                    total_q_values = rewards_v + (GAMMA * next_q_values_max)

                action = int(torch.argmax(total_q_values).item())

            next_state_dict, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            memory.push(state_dict, action, reward, next_state_dict, float(done))

            state_dict = next_state_dict
            total_reward += reward
            steps += 1
            frame_idx += 1

            loss = optimize_model(memory, policy_net, target_net, optimizer)

        if ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Ep {ep:5d} | reward {total_reward:8.2f} | steps {steps:4d} | eps {eps:.4f} | mem {len(memory)}")

        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(policy_net.state_dict(), BEST_MODEL_FILE)
            print(f"*** New best reward: {best_reward:.2f}, best model saved. ***")

        if ep % 500 == 0:
            print("--- Saving RL checkpoint... ---")
            torch.save({
                'epoch': ep, 'frame_idx': frame_idx,
                'policy_net_state_dict': policy_net.state_dict(),
                'target_net_state_dict': target_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'memory': memory, 'best_reward': best_reward,
            }, CHECKPOINT_FILE)

    torch.save(policy_net.state_dict(), "dqn_tetris_final_lookahead_dqn.pth")
    print("Training finished. Best reward:", best_reward)

"""
Trains a Deep Q-Network (DQN) agent to play Tetris.

This agent uses a custom Tetris environment (TetrisEnv) where the action
space represents the final (rotation, column) placement of a piece.

A key feature of this implementation is the use of a one-step lookahead
strategy during action selection (when not exploring). Instead of just
picking the action with the highest Q-value for the *current* state,
the agent:

1.  Simulates all possible next moves using `env.get_next_states()`.
2.  Gets the *actual* reward and 'done' status for each simulated move.
3.  Calculates the Q-value of the *resulting next states* using the
    policy network.
4.  Selects the action that maximizes the total expected future reward:
    `Q_total = (actual_reward + GAMMA * Q_next_state)`.

This grounds the agent's decision in the immediate, deterministic
consequences of its actions, which is highly effective in Tetris.
"""
from collections import deque
import os
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from bots.cnn_training.tetris_env_cnn import TetrisEnv, ROWS, COLS, ORDER


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeepQNetwork(nn.Module):
    """
        Deep Q-Network model for approximating the Q-function.

        This network takes two inputs from the state dictionary:
        1.  'board': (N, 1, ROWS, COLS) - A binary representation of the game board.
        2.  'kind': (N,) - The integer index of the current tetromino piece.

        The 'board' is processed by 2D convolutional layers.
        The 'kind' is one-hot encoded.
        The features are then concatenated and passed through fully connected
        layers to produce Q-values for all possible actions.
    """
    def __init__(self, h=ROWS, w=COLS, num_kinds=len(ORDER), num_actions=4 * COLS):
        """
        Initializes the neural network layers.

        Args:
            h (int): Height of the game board (ROWS).
            w (int): Width of the game board (COLS).
            num_kinds (int): Number of different tetromino types (e.g., 7).
            num_actions (int): Total number of possible actions (e.g., 4*10 = 40).
        """
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

    def forward(self, state_input):
        """
        Performs the forward pass of the network.
        Args:
            state_input (Dict[str, torch.Tensor]): A dictionary containing:
                - 'board': The board tensor.
                - 'kind': The current piece kind tensor.

        Returns:
            torch.Tensor: The predicted Q-values for each action.
        """
        board_tensor = state_input['board']
        kind_tensor = state_input['kind']

        if board_tensor.dim() == 3:
            board_tensor = board_tensor.unsqueeze(1)
        if board_tensor.shape[1] != 1:
            board_tensor = board_tensor.reshape(-1, 1, ROWS, COLS)
        conv_features = self.conv(board_tensor)
        kind_onehot = F.one_hot(kind_tensor.long(), num_classes=self.num_kinds).float()  # pylint: disable=E1102
        if kind_onehot.dim() == 1:
            kind_onehot = kind_onehot.unsqueeze(0)
        combined_features = torch.cat([conv_features, kind_onehot], dim=1)
        shared_output = self.fc_shared(combined_features)
        q = self.fc_output(shared_output)
        return q

class ReplayMemory:
    """
    A simple FIFO replay buffer for storing and sampling experiences.

    This is used in off-policy RL algorithms like DQN to store
    (state, action, reward, next_state, done) transitions.
    """
    def __init__(self, capacity):
        """
        Initializes the replay memory.

        Args:
            capacity (int): The maximum number of transitions to store.
        """
        self.memory = deque(maxlen=capacity)

    def push(self, s_dict, a, r, next_s_dict, d):
        # pylint: disable=R0913
        # pylint: disable=R0917
        """
        Saves a transition to the memory.

        Args:
            s_dict (dict): The state observation.
            a (int): The action taken.
            r (float): The reward received.
            next_s_dict (dict): The resulting next state observation.
            d (bool): Whether the episode terminated.
        """
        self.memory.append((s_dict, a, r, next_s_dict, d))

    def sample(self, batch_size):
        """
        Randomly samples a batch of transitions from the memory.

        Args:
            batch_size (int): The number of transitions to sample.

        Returns:
            tuple: A tuple containing batched data:
                    (states, actions, rewards, next_states, dones).
                    States and next_states are dictionaries of numpy arrays.
        """
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
        """Returns the current number of transitions in the memory."""
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

def epsilon_by_frame(current_frame_idx):
    """
        Calculates the current epsilon value based on a linear decay schedule.

        Args:
            current_frame_idx (int): The current total number of frames (steps) elapsed.

        Returns:
            float: The calculated epsilon value, clipped between EPS_END and EPS_START.
    """
    return max(EPS_END, EPS_START - current_frame_idx * (EPS_START - EPS_END) / EPS_DECAY_FRAMES)


def optimize_model(mem, p_net, t_net, agent_optimizer):
    # pylint: disable=R0914
    """
        Performs one step of optimization on the policy network.

        Samples a batch from 'mem', calculates the loss using the 't_net',
        and updates the 'p_net' using the 'agent_optimizer'.

        Args:
            mem (ReplayMemory): The replay buffer.
            p_net (DeepQNetwork): The main Q-network (being trained).
            t_net (DeepQNetwork): The target Q-network (for stable targets).
            agent_optimizer (optim.Optimizer): The optimizer for the policy network.

        Returns:
            float or None: The loss value if training occurred, otherwise None.
    """
    if len(mem) < BATCH_SIZE:
        return None

    (states_dict, actions, rewards, next_states_dict, dones) = mem.sample(BATCH_SIZE)
    states_board_v = torch.from_numpy(states_dict['board']).float().to(DEVICE)
    states_kind_v = torch.from_numpy(states_dict['kind']).long().to(DEVICE)
    next_states_board_v = torch.from_numpy(next_states_dict['board']).float().to(DEVICE)
    next_states_kind_v = torch.from_numpy(next_states_dict['kind']).long().to(DEVICE)
    actions_v = torch.from_numpy(actions).long().to(DEVICE).unsqueeze(1)
    rewards_v = torch.from_numpy(rewards).float().to(DEVICE)
    dones_v = torch.from_numpy(dones).float().to(DEVICE)

    current_state_input = {'board': states_board_v, 'kind': states_kind_v}
    next_state_input = {'board': next_states_board_v, 'kind': next_states_kind_v}
    q_values = p_net(current_state_input).gather(1, actions_v).squeeze(1)

    with torch.no_grad():
        next_q_values_max = t_net(next_state_input).max(dim=1)[0]
        next_q_values_max = next_q_values_max * (1.0 - dones_v)
        target_q_values = rewards_v + (GAMMA * next_q_values_max)
    loss_fn = nn.MSELoss()
    loss = loss_fn(q_values, target_q_values)
    agent_optimizer.zero_grad()
    loss.backward()
    agent_optimizer.step()

    return loss.item()


if __name__ == '__main__':

    # Main training loop.

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
        print(f"check{CHECKPOINT_FILE}...")
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
        except (FileNotFoundError, KeyError, RuntimeError, EOFError) as e:
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

                lookahead_boards_v = torch.from_numpy(all_next_boards).float().to(DEVICE)
                lookahead_kinds_v = torch.from_numpy(all_next_kinds).long().to(DEVICE)
                lookahead_rewards_v = torch.from_numpy(all_rewards).float().to(DEVICE)
                lookahead_dones_v = torch.from_numpy(all_dones).float().to(DEVICE)

                lookahead_state_input = {'board': lookahead_boards_v, 'kind': lookahead_kinds_v}
                with torch.no_grad():
                    lookahead_next_q_max = policy_net(lookahead_state_input).max(dim=1)[0]
                    lookahead_next_q_max = lookahead_next_q_max * (1.0 - lookahead_dones_v)
                    total_q_values = lookahead_rewards_v + (GAMMA * lookahead_next_q_max)

                action = int(torch.argmax(total_q_values).item())

            next_state_dict, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            memory.push(state_dict, action, reward, next_state_dict, float(done))

            state_dict = next_state_dict
            total_reward += reward
            steps += 1
            frame_idx += 1
            current_loss = optimize_model(memory, policy_net, target_net, optimizer)

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

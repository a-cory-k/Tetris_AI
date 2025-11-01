# pylint: disable=no-member, too-many-locals
"""
LstmBot implementation for playing Tetris.

This bot uses an LSTM model trained on feature vectors derived from
the game state history. It maintains a sequence of the last N states
to predict the next best action.
"""
import torch
import torch.nn as nn
import numpy as np
from app.tetris_dual import Piece, Game
from bots.heuristic_bot import apply_move, Bot_Trainer


class ActorLSTM(nn.Module):
    """
    The ActorLSTM model class, matching the training architecture.
    """

    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0, num_actions=40):
        """
        Initializes the LSTM model layers.

        Args:
            input_dim (int): The number of features in the input vector.
            hidden_dim (int): The size of the LSTM hidden state.
            num_layers (int): The number of LSTM layers.
            dropout (float): Dropout probability (for training).
            num_actions (int): The size of the output action space.
        """
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        """Performs a forward pass through the network."""
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class LstmBot:
    """
    A bot that uses a trained LSTM model to make decisions.
    """

    def __init__(self, model_path, scaler_mean, scaler_scale, seq_len=15, num_rotations=4, num_cols=10):
        """
        Initializes the bot, loads the model, and sets up the scaler.

        Args:
            model_path (str): Path to the .pth model weights file.
            scaler_mean (np.ndarray): The mean_ array from the training StandardScaler.
            scaler_scale (np.ndarray): The scale_ array from the training StandardScaler.
            seq_len (int): The sequence length the LSTM was trained on.
            num_rotations (int): Number of rotations (must match training).
            num_cols (int): Number of columns (must match training).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_rotations = num_rotations
        self.num_cols = num_cols
        self.num_actions = num_rotations * num_cols
        self.seq_len = seq_len

        self.num_cols_list = ['holes', 'bumpiness', 'cleared', 'aggregate_height', 'max_height', 'holes_density',
                              'surface_roughness']

        self.input_dim = 10 + len(self.num_cols_list) + 7

        self.scaler_mean = np.array(scaler_mean)
        self.scaler_scale = np.array(scaler_scale)

        self.model = ActorLSTM(input_dim=self.input_dim, num_actions=self.num_actions).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.kind_map = {'I': 0, 'J': 1, 'L': 2, 'O': 3, 'S': 4, 'T': 5, 'Z': 6}

        self.history = []

        self.heuristic_bot = Bot_Trainer(None)

        print(f"LstmBot initialized. Input dim: {self.input_dim}, Device: {self.device}")

    def reset(self):
        """Resets the bot's state history for a new game."""
        self.history = []

    def _get_board_properties(self, game):
        """
        Extracts numerical board properties (heights, holes, etc.).
        """
        board = game.board

        grid = board.create_grid()

        board_height = len(grid)
        board_width = len(grid[0]) if board_height > 0 else 0

        heights = [0] * board_width
        for x in range(board_width):
            for y in range(board_height):
                if grid[y][x]:
                    heights[x] = board_height - y
                    break

        heights_arr = np.array(heights)

        properties = {
            'heights': heights_arr / 20.0,
            'holes': 0,
            'bumpiness': 0,
            'cleared': 0,
            'aggregate_height': np.sum(heights_arr),
            'max_height': np.max(heights_arr),
            'holes_density': 0.0,
            'surface_roughness': 0.0,
        }

        for x in range(board_width):
            is_hole = False
            for y in range(board_height):
                if grid[y][x]:
                    is_hole = True
                elif is_hole:
                    properties['holes'] += 1

            if x > 0:
                properties['bumpiness'] += abs(heights[x] - heights[x - 1])

        # TODO: Implement 'holes_density' and 'surface_roughness' if used in training

        return properties

    def _get_features(self, game):
        """
        Assembles the complete feature vector for a single game state.
        """
        props = self._get_board_properties(game)

        heights_features = props['heights']
        if len(heights_features) != 10:
            new_heights = np.zeros(10)
            l = min(len(heights_features), 10)
            new_heights[:l] = heights_features[:l]
            heights_features = new_heights

        num_features_list = [props[col] for col in self.num_cols_list]
        num_features_scaled = (np.array(num_features_list) - self.scaler_mean) / self.scaler_scale

        kind_idx = self.kind_map.get(game.current.kind, 4)
        kind_features = np.zeros(7)
        kind_features[kind_idx] = 1.0

        return np.concatenate([heights_features, num_features_scaled, kind_features]).astype(np.float32)

    def best_move(self, game):
        """
        Calculates and executes the best move based on the LSTM prediction.
        """
        current_features = self._get_features(game)

        self.history.append(current_features)
        if len(self.history) > self.seq_len:
            self.history.pop(0)

        history_tensor_data = []
        if len(self.history) < self.seq_len:
            padding = [current_features] * (self.seq_len - len(self.history))
            history_tensor_data = padding + self.history
        else:
            history_tensor_data = self.history

        history_tensor = torch.tensor(
            np.array([history_tensor_data]),
            dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(history_tensor)
            action_pred = logits.argmax(1).item()

        pred_rot = action_pred % self.num_rotations
        pred_x = action_pred // self.num_rotations

        current_piece_obj = game.current
        best_piece = Piece(pred_x, 0, current_piece_obj.kind)

        num_valid_rots = len(best_piece.shape)
        safe_rot = pred_rot % num_valid_rots
        best_piece.rot = safe_rot

        while game.board.valid(best_piece):
            best_piece.y += 1
        best_piece.y -= 1

        if not game.board.valid(best_piece):
            found_valid = False
            for dx in range(-3, 4):
                test_piece = Piece(best_piece.x + dx, 0, best_piece.kind)
                test_piece.rot = best_piece.rot
                while game.board.valid(test_piece):
                    test_piece.y += 1
                test_piece.y -= 1
                if game.board.valid(test_piece):
                    best_piece = test_piece
                    found_valid = True
                    break

            if not found_valid:
                best_piece_heuristic = self.heuristic_bot.best_move(game)[0]
                if best_piece_heuristic:
                    best_piece = best_piece_heuristic
                else:
                    apply_move(game, game.current)
                    return None, 0

        apply_move(game, best_piece)

        return None, 0


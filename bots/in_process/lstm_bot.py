"""
LSTM Bot implementation for playing Tetris.

This module defines the LSTM model (ActorLSTM) and the bot logic (LSTMBot)
that uses the model to select moves.
"""

import torch
from torch import nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from app.tetris_dual import Piece, COLS, ROWS


class ActorLSTM(nn.Module):
    """LSTM Actor model for policy network."""

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0, num_actions=40):
        """
        Initialize the LSTM Actor model.

        Args:
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Dimension of the LSTM hidden state.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout probability for LSTM.
            num_actions (int): Number of possible output actions.
        """
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        """Forward pass through the LSTM network."""
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class LSTMBot:
    """Bot implementation using the LSTM Actor model."""

    def __init__(self, model_path, input_dim=20+7+7, num_actions=40, scaler_path=None):
        """
        Initialize the LSTMBot.

        Args:
            model_path (str): Path to the saved model state dictionary.
            input_dim (int): Dimension of the input features.
            num_actions (int): Number of possible output actions.
            scaler_path (str, optional): Path to a saved StandardScaler object.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorLSTM(input_dim=input_dim, num_actions=num_actions).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        if scaler_path:
            self.scaler = torch.load(scaler_path)
        else:
            self.scaler = StandardScaler()
            self.scaler.mean_ = np.zeros(input_dim)
            self.scaler.scale_ = np.ones(input_dim)

    def board_to_features(self, game):
        """
        Converts the game board state into a feature vector.

        Args:
            game: The Tetris game state object.

        Returns:
            np.ndarray: A normalized feature vector.
        """
        grid = game.board.create_grid()
        heights = np.array([ROWS - np.argmax([row[x] is not None for row in grid][::-1])
                            if any(row[x] for row in grid) else 0
                            for x in range(COLS)], dtype=np.float32) / 20.0
        holes = sum(1 for x in range(COLS) for y in range(ROWS)
                    if grid[y][x] is None and any(grid[ry][x] for ry in range(y+1, ROWS)))
        aggregate_height = heights.sum()
        max_height = heights.max()
        bumpiness = np.sum(np.abs(np.diff(heights)))
        cleared = game.board.cleared_lines
        holes_density = holes / COLS
        surface_roughness = bumpiness / COLS

        num_feats = np.array([
            holes, bumpiness, cleared, aggregate_height,
            max_height, holes_density, surface_roughness
        ], dtype=np.float32)

        kind_map = {'I': 0, 'J': 1, 'L': 2, 'O': 3, 'S': 4, 'T': 5, 'Z': 6}
        kind_onehot = np.zeros(7, dtype=np.float32)
        kind_onehot[kind_map[game.current.kind]] = 1.0

        feats = np.hstack([heights, num_feats, kind_onehot])
        feats = self.scaler.transform([feats])[0]
        return feats

    def best_move(self, game, seq_len=15):
        """
        Calculates and executes the best move based on the model's policy.

        Args:
            game: The Tetris game state object.
            seq_len (int): The sequence length required by the LSTM.
        """
        feats = self.board_to_features(game)
        X = np.tile(feats, (seq_len, 1))[None, :, :]
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(X_tensor)
            action = logits.argmax(1).item()
        rot = action % 4
        x_pos = action // 4

        best_piece = Piece(x_pos, game.current.y, game.current.kind)
        best_piece.rot = rot
        while game.board.valid(best_piece):
            best_piece.y += 1
        best_piece.y -= 1

        if game.board.valid(best_piece):
            game.current = best_piece
            game.hard_drop()

"""
CNN Bot implementation for playing Tetris.

This module defines the CNN model (CNNWithKind) and the bot logic (CnnBot)
that uses the model to select moves based on board state and current piece.
"""
import torch
import numpy as np
from torch import nn
from app.tetris_dual import Piece
from bots.heuristic_bot import apply_move
class CNNWithKind(nn.Module):
    """A CNN model that takes both the board state and the current piece kind."""
    def __init__(self):
        """Initializes the CNN layers."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_shared = nn.Sequential(
            nn.Linear(64 * 20 * 10 + 7, 256),
            nn.ReLU()
        )
        self.fc_x = nn.Linear(256, 10)
        self.fc_rot = nn.Linear(256, 4)

    def forward(self, x, kind):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): The board state tensor (Batch, 1, 20, 10).
            kind (torch.Tensor): The current piece kind tensor (Batch, 1).

        Returns:
            tuple(torch.Tensor, torch.Tensor): Logits for x-position and rotation.
        """
        x = self.conv(x)
        kind_onehot = nn.functional.one_hot(kind, num_classes=7).float()  # pylint: disable=E1102
        x = torch.cat([x, kind_onehot], dim=1)
        x = self.fc_shared(x)
        out_x = self.fc_x(x)
        out_rot = self.fc_rot(x)
        return out_x, out_rot


class CnnBot:
    """Bot implementation using the CNNWithKind model."""
    def __init__(self, model_path):
        """
        Initializes the bot and loads the trained model.

        Args:
            model_path (str): Path to the saved model state dictionary.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNWithKind().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def board_to_tensor(self, board):
        """
        Converts a Tetris board object into a tensor for the model.

        Args:
            board: The game board object.

        Returns:
            torch.Tensor: A (1, 1, 20, 10) tensor representing the board.
        """
        grid = board.create_grid()
        arr = np.zeros((1, 20, 10), dtype=np.float32)
        for y in range(20):
            for x in range(10):
                arr[0, y, x] = 1.0 if grid[y][x] else 0.0
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(self.device)

    def best_move(self, game):
        """
        Calculates and executes the best move based on the model's prediction.

        Args:
            game: The main Tetris game state object.
        """
        current_piece = game.current
        board_tensor = self.board_to_tensor(game.board)
        kind_map = {'I': 0, 'J': 1, 'L': 2, 'O': 3, 'S': 4, 'T': 5, 'Z': 6}
        kind_tensor = torch.tensor([kind_map[current_piece.kind]], dtype=torch.long).to(self.device)

        with torch.no_grad():
            out_x, out_rot = self.model(board_tensor, kind_tensor)
            x_pred = out_x.argmax(1).item()
            rot_pred = out_rot.argmax(1).item()


        best_piece = Piece(x_pred, current_piece.y, current_piece.kind)
        best_piece.rot = rot_pred
        while game.board.valid(best_piece):
            best_piece.y += 1
        best_piece.y -= 1


        if not game.board.valid(best_piece):
            for dx in range(-3, 4):
                test_piece = Piece(best_piece.x + dx, best_piece.y, best_piece.kind)
                test_piece.rot = best_piece.rot
                while game.board.valid(test_piece):
                    test_piece.y += 1
                test_piece.y -= 1
                if game.board.valid(test_piece):
                    best_piece = test_piece
                    break

        apply_move(game, best_piece)

        return None, 0

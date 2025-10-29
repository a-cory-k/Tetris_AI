import pytest
import sys
import numpy as np
from unittest.mock import MagicMock, patch

mock_pygame = MagicMock()
mock_pygame.font = MagicMock()
mock_pygame.font.SysFont = MagicMock(return_value=MagicMock())

sys.modules["pygame"] = mock_pygame
sys.modules["pygame.font"] = mock_pygame.font

mock_torch = MagicMock()
mock_torch.nn = MagicMock()
mock_torch.nn.Module = MagicMock
mock_torch.nn.Sequential = MagicMock
mock_torch.nn.Conv2d = MagicMock
mock_torch.nn.ReLU = MagicMock
mock_torch.nn.Flatten = MagicMock
mock_torch.nn.Linear = MagicMock
mock_torch.nn.functional = MagicMock
mock_torch.cuda = MagicMock()
mock_torch.device = MagicMock(return_value="cpu")
mock_torch.load = MagicMock(return_value=MagicMock())

mock_torch.no_grad = MagicMock() 
mock_torch.no_grad.return_value.__enter__ = MagicMock()
mock_torch.no_grad.return_value.__exit__ = MagicMock()

mock_torch.tensor = MagicMock(return_value=MagicMock(to=MagicMock()))

sys.modules["torch"] = mock_torch
sys.modules["torch.nn"] = mock_torch.nn

from app.tetris_dual import Piece, Board, Game, COLS, ROWS
from bots.cnn_bot import CnnBot

@pytest.fixture
def game():
    """Provides a clean, new Game object for each test."""
    return Game()

@pytest.fixture
def board():
    """Provides a clean, new Board object for each test."""
    return Board()

@pytest.fixture
def mocked_bot(mocker):
    """
    Provides a CnnBot instance with all torch dependencies mocked.
    
    This is the primary fixture for these tests. It patches torch,
    the model (CNNWithKind), and file loading.
    """
    # --- ИСПРАВЛЕНИЕ ПУТЕЙ ЗДЕСЬ ---
    mocker.patch('bots.cnn_bot.torch.cuda.is_available', return_value=False) # <-- Было app.cnn_bot
    
    mock_model_instance = MagicMock()
    mock_model_instance.to.return_value = mock_model_instance
    mocker.patch('bots.cnn_bot.CNNWithKind', return_value=mock_model_instance) # <-- Было app.cnn_bot
    
    mocker.patch('bots.cnn_bot.torch.load', return_value=MagicMock()) # <-- Было app.cnn_bot
    mocker.patch('bots.cnn_bot.torch.tensor', return_value=MagicMock(to=MagicMock(unsqueeze=MagicMock(return_value="mock_tensor")))) # <-- Было app.cnn_bot
    # --- КОНЕЦ ИСПРАВЛЕНИЙ ---

    bot = CnnBot("fake/path.pth")
    
    return bot, mock_model_instance


class TestCnnBotInit:
    """Tests the CnnBot constructor (__init__)."""

    def test_init_loads_model(self, mocked_bot):
        """Tests that __init__ correctly loads the model and sets it to eval mode."""
        bot, mock_model = mocked_bot
        
        mock_torch.device.assert_called_with("cpu")
        
        # Проверяем, что мок-версия CNNWithKind была вызвана
        assert bot.model == mock_model
        
        mock_torch.load.assert_called_with("fake/path.pth", map_location="cpu")
        mock_model.load_state_dict.assert_called()
        
        mock_model.eval.assert_called()


class TestBoardToTensor:
    """Tests the board_to_tensor conversion logic."""

    # --- ИСПРАВЛЕНИЕ ПУТИ ЗДЕСЬ ---
    @patch('bots.cnn_bot.torch.tensor') # <-- Было app.cnn_bot
    def test_board_to_tensor_conversion(self, mock_tensor_func, board):
        """Tests that the board array is correctly converted to a numpy array."""
        
        bot = CnnBot.__new__(CnnBot)
        bot.device = "cpu"
        
        mock_tensor_obj = MagicMock()
        mock_tensor_func.return_value = mock_tensor_obj
        mock_tensor_obj.unsqueeze.return_value.to.return_value = "final_tensor"

        board.locked[(5, 19)] = (1, 1, 1)
        
        result = bot.board_to_tensor(board)

        assert result == "final_tensor"
        
        assert mock_tensor_func.called
        call_args = mock_tensor_func.call_args[0][0]
        
        assert isinstance(call_args, np.ndarray)
        assert call_args.shape == (1, 20, 10)
        assert call_args[0, 19, 5] == 1.0
        assert call_args[0, 0, 0] == 0.0

class TestBestMove:
    """Tests the bot's best_move logic."""

    def test_best_move_follows_model_prediction(self, mocked_bot, game):
        """Tests that the bot correctly applies the move predicted by the mocked model."""
        bot, mock_model = mocked_bot
        
        mock_output_x = MagicMock()
        mock_output_rot = MagicMock()
        
        mock_output_x.argmax.return_value.item.return_value = 5
        mock_output_rot.argmax.return_value.item.return_value = 1
        
        mock_model.return_value = (mock_output_x, mock_output_rot)

        game.current = Piece(COLS // 2, 0, 'T')
        
        bot.best_move(game)

        expected_cells = {
            (5, 17),
            (5, 18),
            (6, 18),
            (5, 19)
        }
        
        assert len(game.board.locked) == 4
        assert set(game.board.locked.keys()) == expected_cells


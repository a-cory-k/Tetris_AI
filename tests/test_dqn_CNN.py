"""
Unit tests for the DQNBot class.

This test suite mocks all external dependencies (torch, numpy, pygame,
and custom training/environment modules) to test the bot's
logic in isolation.
"""

import sys
from unittest.mock import MagicMock
import pytest
import numpy as np

mock_pygame = MagicMock()
mock_pygame.font = MagicMock()
mock_pygame.font.SysFont = MagicMock(return_value=MagicMock())
sys.modules["pygame"] = mock_pygame
sys.modules["pygame.font"] = mock_pygame.font


mock_torch = MagicMock()
mock_torch.nn = MagicMock()
mock_torch.nn.Module = MagicMock
mock_torch.device = MagicMock(return_value="cpu")
mock_torch.load = MagicMock(return_value=MagicMock())
mock_torch.no_grad = MagicMock()
mock_torch.no_grad.return_value.__enter__ = MagicMock()
mock_torch.no_grad.return_value.__exit__ = MagicMock()
mock_torch.from_numpy = MagicMock(
    return_value=MagicMock(float=MagicMock(return_value=MagicMock(to=MagicMock())))
)
mock_torch.argmax = MagicMock(return_value=MagicMock(item=MagicMock()))
sys.modules["torch"] = mock_torch
sys.modules["torch.nn"] = mock_torch.nn

mock_tetris_env = MagicMock()
mock_tetris_env.TetrisEnv = MagicMock()
sys.modules["bots.cnn_training.tetris_env_cnn"] = mock_tetris_env

mock_tetris_train = MagicMock()
mock_tetris_train.DeepQNetwork = MagicMock()
mock_tetris_train.DEVICE = "cpu"
mock_tetris_train.GAMMA = 0.99
sys.modules["bots.cnn_training.tetris_train_cnn"] = mock_tetris_train

from app.tetris_dual import Game, Piece
from bots.dqn_bot_CNN import DQNBot
from bots.cnn_training.tetris_env_cnn import TetrisEnv
from bots.cnn_training.tetris_train_cnn import DeepQNetwork


@pytest.fixture
def game():
    """Provides a clean, new Game object for each test."""
    return Game()

@pytest.fixture
def mocked_bot(mocker):
    """
    Provides a DQNBot instance with all dependencies mocked.
    """
    mock_env_instance = MagicMock()
    mock_env_instance.action_space.n = 40
    mock_env_instance.get_next_states = MagicMock()
    mocker.patch.object(TetrisEnv, '__call__', return_value=mock_env_instance)
    mocker.patch('bots.dqn_bot_CNN.TetrisEnv', return_value=mock_env_instance)

    mock_model_instance = MagicMock()
    mock_model_instance.to.return_value = mock_model_instance
    mock_model_instance.load_state_dict = MagicMock()
    mock_model_instance.eval = MagicMock()
    mock_model_instance.return_value = MagicMock()

    DeepQNetwork.return_value = mock_model_instance

    mocker.patch('bots.dqn_bot_CNN.torch.load')

    bot = DQNBot("fake/path.pth")

    return bot, mock_model_instance, mock_env_instance


class TestDQNBotInit:
    """Tests the DQNBot constructor."""

    def test_init_loads_model(self, mocked_bot):
        """
        Tests that the __init__ method correctly initializes the adapter
        environment, instantiates the DeepQNetwork, loads the state
        dictionary, and sets the model to evaluation mode.
        """
        bot, mock_model, mock_env = mocked_bot

        assert bot.adapter_env == mock_env
        assert DeepQNetwork.called
        assert bot.agent_model == mock_model
        assert mock_model.to.called
        assert mock_model.load_state_dict.called
        assert mock_model.eval.called

    def test_init_handles_load_error(self, mocker, capsys):
        """
        Tests that the bot handles a FileNotFoundError or RuntimeError
        during model loading by printing a warning and continuing.
        """
        mocker.patch('bots.dqn_bot_CNN.TetrisEnv', return_value=MagicMock(action_space=MagicMock(n=40)))

        mock_model_instance = MagicMock(to=MagicMock())
        DeepQNetwork.return_value = mock_model_instance

        mocker.patch('bots.dqn_bot_CNN.torch.load', side_effect=FileNotFoundError("Test error"))

        DQNBot("fake/path.pth")

        captured = capsys.readouterr()
        assert "Warning: Could not load model" in captured.out
        assert "Test error" in captured.out


def get_data(operand):
    """
    Safely extracts the .data attribute from a mock tensor,
    or returns the operand if it's a scalar (e.g., float, int).
    """
    return operand.data if hasattr(operand, 'data') else operand

class TestBestMove:
    """Tests the best_move logic of the DQNBot."""

    def test_best_move_no_states(self, mocked_bot, game, mocker):
        """
        Tests that the bot performs a hard_drop if the environment
        returns no possible next states.
        """
        bot, mock_model, mock_env = mocked_bot

        mock_env.get_next_states.return_value = ({}, None)

        mocker.patch.object(game, 'hard_drop')

        bot.best_move(game)

        game.hard_drop.assert_called_once()
        mock_model.assert_not_called()

    def test_best_move_selects_best_q_value(self, mocked_bot, game, mocker):
        """
        Tests that the bot correctly identifies the action with the
        highest calculated Q-value and applies that move to the game.
        """
        bot, mock_model, mock_env = mocked_bot

        next_states_info = {
            5:  ({'board': "board_data_5", 'kind': 0}, 10.0, False),
            11: ({'board': "board_data_11", 'kind': 0}, 20.0, False)
        }
        mock_env.get_next_states.return_value = (next_states_info, None)

        mocker.patch('bots.dqn_bot_CNN.np.concatenate', return_value="boards_concat")

        original_np_array = np.array
        mocker.patch('bots.dqn_bot_CNN.np.array',
                     side_effect=lambda x, **kwargs: original_np_array(x))

        def tensor_factory(data):
            """A factory to create mock tensors that support operations."""
            data = np.array(data)
            mock_tensor = MagicMock(name=f"tensor_{data}")
            mock_tensor.data = data
            mock_tensor.float.return_value = mock_tensor
            mock_tensor.long.return_value = mock_tensor
            mock_tensor.to.return_value = mock_tensor

            mock_tensor.__add__ = lambda self, other: tensor_factory(self.data + get_data(other))
            mock_tensor.__mul__ = lambda self, other: tensor_factory(self.data * get_data(other))
            mock_tensor.__rmul__ = lambda self, other: tensor_factory(get_data(other) * self.data)
            mock_tensor.__rsub__ = lambda self, other: tensor_factory(get_data(other) - self.data)

            return mock_tensor

        mocker.patch('bots.dqn_bot_CNN.torch.from_numpy', side_effect=tensor_factory)

        mock_next_q_values = MagicMock()
        mock_model.return_value = mock_next_q_values

        mock_q_max = tensor_factory(np.array([100.0, 50.0]))
        mock_next_q_values.max.return_value = (mock_q_max,)

        mock_argmax_item = MagicMock()
        mock_argmax_item.item.return_value = 0
        mocker.patch('bots.dqn_bot_CNN.torch.argmax', return_value=mock_argmax_item)

        game.current = Piece(5, 0, 'T')
        mocker.patch.object(game, 'hard_drop')
        mocker.patch.object(game.board, 'valid', return_value=True)

        def mock_rotated_absolute(x):
            """
            Simulates rotation by creating a new shape and *directly*
            setting its .rot to `x`.
            """
            p = Piece(game.current.x, game.current.y, game.current.kind)
            p.rot = x
            return p

        mocker.patch.object(
            game.current,
            'rotated',
            side_effect=mock_rotated_absolute
        )
        bot.best_move(game)

        game.hard_drop.assert_called_once()
        assert game.current.rot == 0
        assert game.current.x == 0
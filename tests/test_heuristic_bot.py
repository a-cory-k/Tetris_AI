"""testing heuristic bot"""
import pytest
import sys
from unittest.mock import MagicMock

mock_pygame = MagicMock()
mock_pygame.font = MagicMock()
mock_pygame.font.SysFont = MagicMock(return_value=MagicMock())

sys.modules["pygame"] = mock_pygame
sys.modules["pygame.font"] = mock_pygame.font

from app.tetris_dual import Piece, Board, Game, COLS
from bots.heuristic_bot import Bot_Trainer

@pytest.fixture
def bot():
    """Returns a new Bot_Trainer instance for each test."""
    return Bot_Trainer()


@pytest.fixture
def game():
    """Returns a new Game instance for each test."""
    return Game()

@pytest.fixture
def board():
    """Returns a new, empty Board for each test."""
    return Board()

class TestEvaluateBoard:
    """Tests the core heuristic calculations."""

    def test_evaluate_empty_board(self, bot, board):
        """Tests that an empty board results in a score of 0."""
        score, metrics = bot.evaluate_board(board, 0)

        assert score == 0.0
        assert metrics['holes'] == 0
        assert metrics['bumpiness'] == 0
        assert metrics['aggregate_height'] == 0
        assert metrics['cleared'] == 0

    def test_evaluate_one_block(self, bot, board):
        """Tests evaluation of a board with a single block."""
        board.locked[(5, 19)] = (1, 1, 1)

        score, metrics = bot.evaluate_board(board, 0)

        assert metrics['aggregate_height'] == 1
        assert metrics['holes'] == 0
        assert metrics['bumpiness'] == 2
        assert score == pytest.approx(-0.9)

    def test_evaluate_complex_board(self, bot, board):
        """Tests a more complex board with holes, height, and bumpiness."""
        board.locked[(0, 19)] = (1, 1, 1)  # height 1
        board.locked[(1, 18)] = (1, 1, 1)  # height 2
        board.locked[(1, 19)] = (1, 1, 1)
        board.locked[(2, 17)] = (1, 1, 1)  # height 3
        board.locked[(2, 19)] = (1, 1, 1)  # hole at (2, 18)


        score, metrics = bot.evaluate_board(board, 0)


        assert metrics['aggregate_height'] == 6
        assert metrics['holes'] == 1
        assert metrics['bumpiness'] == 5
        assert score == pytest.approx(-4.3)

    def test_evaluate_line_clear_bonus(self, bot, board):
        """Tests that cleared rows add a positive score."""
        score, metrics = bot.evaluate_board(board, 1)
        assert metrics['aggregate_height'] == 0
        assert metrics['holes'] == 0
        assert score == pytest.approx(0.7)
        score, metrics = bot.evaluate_board(board, 4)
        assert score == pytest.approx(2.8)


# Tests for Bot Integration (best_move)

class TestBotIntegration:
    """Tests the bot's ability to interact with the Game object."""

    def test_best_move_simple_drop(self, bot, game):
        """
        Tests if the bot places the 'T' piece on the far left,
        which correctly minimizes bumpiness.
        """
        game.current = Piece(COLS // 2, 0, 'T')  # T-piece

        bot.best_move(game)

        expected_cells = {
            (1, 18),  # .#.
            (0, 19),  # ###
            (1, 19),
            (2, 19)
        }

        assert len(game.board.locked) == 4
        assert set(game.board.locked.keys()) == expected_cells

    def test_best_move_completes_line(self, bot, game):
        """
        Tests if the bot correctly chooses a move that clears a line,
        and that the remaining blocks are correctly shifted.
        """
        for x in range(1, COLS):
            game.board.locked[(x, 19)] = (1, 1, 1)
        game.current = Piece(0, 0, 'I')
        game.current.rot = 1

        bot.best_move(game)

        expected_locked_cells = {
            (0, 17),  # (0, 16) shifted down by 1
            (0, 18),  # (0, 17) shifted down by 1
            (0, 19)  # (0, 18) shifted down by 1
        }

        assert len(game.board.locked) == 3
        assert set(game.board.locked.keys()) == expected_locked_cells
        assert game.lines == 1


"""testing tetris dual """
import pytest
import sys
from unittest.mock import MagicMock

mock_pygame = MagicMock()
mock_pygame.font = MagicMock()
mock_pygame.font.SysFont = MagicMock(return_value=MagicMock())

sys.modules["pygame"] = mock_pygame
sys.modules["pygame.font"] = mock_pygame.font

from app.tetris_dual import Piece, Board, Game, COLS, ROWS, ORDER, COLORS
@pytest.fixture
def board():
    """Returns a new, empty Board for each test."""
    return Board()

@pytest.fixture
def game():
    """Returns a new Game instance for each test."""
    return Game()
class TestPiece:
    """Tests for the Piece class."""

    def test_creation(self):
        """Tests if a 'T' piece is created with correct attributes."""
        p = Piece(5, 0, 'T')
        assert p.x == 5
        assert p.y == 0
        assert p.kind == 'T'
        assert p.rot == 0
        assert p.color == COLORS[ORDER.index('T')]

    def test_cells_T_rot0(self):
        """Tests the cell coordinates for a 'T' piece at rotation 0."""
        p = Piece(5, 0, 'T')
        expected = {(5, -1), (4, 0), (5, 0), (6, 0)}
        assert set(p.cells()) == expected

    def test_cells_I_rot1(self):
        """Tests cell coordinates for an 'I' piece at rotation 1 (vertical)."""
        p = Piece(5, 0, 'I')
        p.rot = 1
        expected = {(5, -1), (5, 0), (5, 1), (5, 2)}
        assert set(p.cells()) == expected

    def test_rotated(self):
        """Tests that rotated() creates a new piece with updated rotation."""
        p = Piece(5, 0, 'T')
        assert p.rot == 0

        p_rot1 = p.rotated(1)
        assert p_rot1.rot == 1
        assert p.rot == 0

        p_rot_neg = p.rotated(-1)
        assert p_rot_neg.rot == 3
class TestBoard:
    """Tests for the Board class."""

    def test_valid_empty(self, board):
        """Tests if a piece is valid on an empty board."""
        p = Piece(COLS // 2, 0, 'T')
        assert board.valid(p) is True

    def test_valid_out_of_bounds_left(self, board):
        """Tests if a piece is invalid when placed off the left boundary."""
        p_left = Piece(0, 0, 'T')
        assert board.valid(p_left) is False

    def test_valid_out_of_bounds_right(self, board):
        """Tests if a piece is invalid when placed off the right boundary."""
        p_right = Piece(9, 0, 'T')
        assert board.valid(p_right) is False

    def test_valid_out_of_bounds_bottom(self, board):
        """Tests if a piece is invalid when placed off the bottom boundary."""
        p_overflow = Piece(5, ROWS, 'T')
        assert board.valid(p_overflow) is False

    def test_valid_collision(self, board):
        """Tests if a piece is invalid when colliding with locked cells."""
        board.locked[(5, 5)] = (255, 0, 0)
        p = Piece(4, 5, 'O')
        assert board.valid(p) is False

    def test_add_piece(self, board):
        """Tests that add_piece() correctly locks the piece's cells onto the board."""
        p = Piece(5, 0, 'T')
        cells = p.cells()
        board.add_piece(p)
        assert len(board.locked) == 4
        for cell in cells:
            assert cell in board.locked
            assert board.locked[cell] == p.color

    def test_clear_one_row(self, board):
        """Tests clearing a single completed row and shifting blocks down."""
        for x in range(COLS):
            board.locked[(x, 19)] = (1, 1, 1)
        board.locked[(5, 18)] = (2, 2, 2)

        cleared_count = board.clear_rows()

        assert cleared_count == 1
        assert board.cleared_lines == 1
        assert (5, 18) not in board.locked
        assert (5, 19) in board.locked
        assert board.locked[(5, 19)] == (2, 2, 2)
        assert (0, 19) not in board.locked
    def test_clear_tetris(self, board):
        """Tests clearing four rows (a Tetris) and shifting blocks down."""
        # Fill rows 16, 17, 18, 19
        for y in [16, 17, 18, 19]:
            for x in range(COLS):
                board.locked[(x, y)] = (1, 1, 1)
        # Add a block above
        board.locked[(5, 15)] = (2, 2, 2)

        cleared_count = board.clear_rows()

        assert cleared_count == 4
        assert board.cleared_lines == 4
        assert (5, 15) not in board.locked
        assert (5, 19) in board.locked  # Block shifted down 4 rows
        assert board.locked[(5, 19)] == (2, 2, 2)

    def test_check_lost(self, board):
        """Tests check_lost() returns True if a piece is locked above the playfield."""
        board.locked[(5, -1)] = (1, 1, 1)  # Block above screen
        assert board.check_lost() is True

    def test_check_not_lost(self, board):
        """Tests check_lost() returns False if all locked pieces are within the playfield."""
        board.locked[(5, 0)] = (1, 1, 1)
        assert board.check_lost() is False

class TestGame:
    """Tests for the main Game logic class."""

    def test_game_creation(self, game):
        """Tests if the game initializes with correct default values."""
        assert isinstance(game.board, Board)
        assert isinstance(game.current, Piece)
        assert len(game.queue) > 0
        assert len(game.next_piece) == 3
        assert game.score == 0
        assert game.lines == 0
        assert game.over is False
        assert game.paused is False

    def test_new_bag(self, game):
        """Tests that new_bag() creates a shuffled list of all 7 piece types."""
        bag = game.new_bag()
        assert len(bag) == len(ORDER)
        assert set(bag) == set(ORDER)

    def test_get_next_refills(self, game):
        """Tests that get_next() refills the queue if it becomes too short."""
        game.queue = ['T']  # < 3, should refill
        p = game.get_next()
        assert p.kind == 'T'
        assert len(game.queue) == len(ORDER)  # 7 (a new bag)

    def test_peek_next_refills(self, game):
        """Tests that peek_next() refills the queue if it's too short to peek."""
        game.queue = ['T', 'I']  # < 3
        peeked = game.peek_next(3)
        assert len(peeked) == 3
        assert peeked[0] == 'T'
        assert peeked[1] == 'I'
        assert len(game.queue) == 2 + len(ORDER)  # 9

    def test_lock_piece_scoring(self, game):
        """Tests that lock_piece() correctly calculates score for a single line clear."""
        # Manually fill a row
        game.board = Board()
        for x in range(COLS):
            game.board.locked[(x, 19)] = (1, 1, 1)
        game.current = Piece(5, 0, 'T')  # Any piece to trigger the lock
        old_piece = game.current

        game.lock_piece()

        assert game.lines == 1
        assert game.score == 100
        assert game.current != old_piece  # Got a new piece
        assert len(game.board.locked) == 4  # Old piece was locked

    def test_game_over(self, game):
        """Tests if the game ends when a new piece spawns in an occupied space."""
        # Lock a cell in the spawn area
        game.board.locked[(COLS // 2, 0)] = (1, 1, 1)  # (5, 0)

        # lock_piece() will spawn a new 'T' at (5, -1)
        # Its cells: (5, -1), (4, 0), (5, 0), (6, 0)
        # It will collide with (5, 0)
        game.lock_piece()
        assert game.over is True

    def test_hard_drop(self, game):
        """Tests that hard_drop() drops the piece to the bottom and locks it."""
        game.current = Piece(COLS // 2, 0, 'T')
        old_piece = game.current

        game.hard_drop()
        assert (5, 18) in game.board.locked
        assert (4, 19) in game.board.locked
        assert (5, 19) in game.board.locked
        assert (6, 19) in game.board.locked
        assert game.current != old_piece

    def test_step_gravity(self, game):
        """Tests that step() moves the piece down when gravity interval passes."""
        game.current = Piece(COLS // 2, 0, 'T')
        game.fall_interval = 0.5

        game.step(0.2)
        assert game.current.y == 0
        assert game.fall_time == 0.2

        game.step(0.4)
        assert game.current.y == 1
        assert game.fall_time == 0.0
    def test_step_soft_drop(self, game):
        """Tests that step() moves the piece down faster during a soft drop."""
        game.current = Piece(COLS // 2, 0, 'T')
        game.fall_interval = 0.5
        game.soft_drop = True
        game.step(0.2)
        assert game.current.y == 1
        assert game.fall_time == 0.0  # Reset

    def test_step_locks_at_bottom(self, game):
        """Tests that step() locks the piece when it tries to move past the bottom."""
        game.current = Piece(COLS // 2, 19, 'T')
        game.fall_interval = 0.5

        game.step(0.6)

        assert (5, 18) in game.board.locked
        assert (4, 19) in game.board.locked
        assert game.current.y < 0

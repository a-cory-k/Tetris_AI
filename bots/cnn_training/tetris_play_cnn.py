"""
Module for running a trained CNN Tetris bot using Pygame. (for testing purposes)
"""
import pygame
import torch
import numpy as np
from app.tetris_dual import App, Game, ORDER, MARGIN
from bots.cnn_training.tetris_env_cnn import TetrisEnv, ROWS, COLS
from bots.cnn_training.tetris_train_cnn import DeepQNetwork, GAMMA, DEVICE


class BotApp(App):
    """
    Pygame application class for running and displaying the Tetris bot's gameplay.
    """

    def __init__(self, model_path):
        super().__init__()
        self.bot_env = TetrisEnv()
        action_dim = self.bot_env.action_space.n
        self.agent_model = DeepQNetwork(h=ROWS, w=COLS, num_kinds=len(ORDER), num_actions=action_dim).to(DEVICE)
        try:
            self.agent_model.load_state_dict(
                torch.load(model_path, map_location=DEVICE))
            print(f"Model successfully loaded from {model_path}")
        except FileNotFoundError:
            print(f"ERROR: Model file not found at: {model_path}")
        except RuntimeError as e:
            print(f"ERROR: Failed to load model: {e}")
        self.agent_model.eval()

    def _choose_action(self):
        """
        Calculates and returns the best action based on the current
        game state using the DQN model.
        """
        self.bot_env.game = self.game
        next_states_info, _ = self.bot_env.get_next_states()

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
            next_q_values_max = self.agent_model(next_state_input).max(dim=1)[0]
            next_q_values_max = next_q_values_max * (1.0 - dones_v)
            total_q_values = rewards_v + (GAMMA * next_q_values_max)
        action = int(torch.argmax(total_q_values).item())
        return action

    def run_bot(self):
        """
        Runs the main game loop for the bot.
        Handles user input (for restart/quit) and bot moves.
        """
        running = True
        self.game = Game()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:  # pylint: disable=no-member
                    running = False
                if event.type == pygame.KEYDOWN:  # pylint: disable=no-member
                    if event.key == pygame.K_r:  # pylint: disable=no-member
                        self.game = Game()
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):  # pylint: disable=no-member
                        running = False
            if not self.game.over:
                action = self._choose_action()
                rot, col = divmod(action, COLS)
                new_piece = self.game.current.rotated(rot)
                new_piece.x = col
                if self.game.board.valid(new_piece):
                    self.game.current = new_piece
                self.game.hard_drop()

            self.renderer.draw_window(self.game, MARGIN, MARGIN)

            if not self.game.over:
                self.renderer.draw_piece(
                    self.game.current, MARGIN, MARGIN)
            else:
                self.renderer.draw_center_text("Bot finished!", sub="R to restart")

            pygame.display.flip()
        pygame.quit()  # pylint: disable=no-member


if __name__ == "__main__":
    app = BotApp("models/tetris_dqn_v2.pth")
    app.run_bot()

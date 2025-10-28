import pygame
import torch
import numpy as np
from tetris_test import App, Game, ORDER
from tetris_env_cnn import TetrisEnv, ROWS, COLS
from tetris_train_cnn import DeepQNetwork, GAMMA, DEVICE


class BotApp(App):
    def __init__(self, model_path):
        super().__init__()
        self.bot_env = TetrisEnv()
        action_dim = self.bot_env.action_space.n
        self.agent_model = DeepQNetwork(h=ROWS, w=COLS, num_kinds=len(ORDER), num_actions=action_dim).to(DEVICE)
        try:
            self.agent_model.load_state_dict(
                torch.load(model_path, map_location=DEVICE))
        except Exception as e:
        self.agent_model.eval()

    def run_bot(self):
        running = True
        self.game = Game()
        while running:
            dt = self.clock.tick(10) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.game = Game()
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
            if not self.game.over:
                self.bot_env.game = self.game
                next_states_info, current_kind_idx = self.bot_env.get_next_states()
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
                    total_q_values = rewards_v + (GAMMA * next_q_values_max)  # 40

                action = int(torch.argmax(total_q_values).item())

                rot, col = divmod(action, COLS)
                new_piece = self.game.current.rotated(rot)
                new_piece.x = col
                if self.game.board.valid(new_piece):
                    self.game.current = new_piece
                self.game.hard_drop()

            self.renderer.draw_window(self.game)
            if not self.game.over:
                self.renderer.draw_piece(
                    self.game.current)
                self.renderer.draw_next(self.game.queue)
            else:
                self.renderer.draw_center_text("Bot finished!", sub="R to restart")
            pygame.display.flip()
        pygame.quit()


if __name__ == "__main__":
    try:
        app = BotApp("dqn_tetris_best_lookahead_dqn.pth")
        app.run_bot()
    except FileNotFoundError:
        print("Ошибка: Файл лучшей модели 'dqn_tetris_final_lookahead_dqn.pth' не найден.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

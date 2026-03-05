# 🎮 Tetris AI
*Research project exploring AI agents in the Tetris environment using Reinforcement Learning and Deep Neural Networks.*

<p align="center">
  <img src="https://img.shields.io/badge/status-Work%20in%20Progress-orange" />
  <img src="https://img.shields.io/badge/AI-DQN%2C%20CNN%2C%20LSTM-blue" />
  <img src="https://img.shields.io/badge/license-MIT-green" />
</p>

---

## 🚀 Overview
This project is an experimental framework for developing and evaluating AI agents that play **Tetris**. The goal is to compare different learning strategies—such as **Deep Q-Networks (DQN)**, **Convolutional Neural Networks (CNN)**, **LSTM-based actors** (in process),**Graph Neural Networks (GNNs)** (in process), and **heuristic-based methods**—in terms of performance, stability, and learning efficiency.

The project is actively under development and serves as a research environment for reinforcement learning, neural architectures, and dataset-driven policy optimization.

---

## 📂 Repository Structure
```plaintext
├── README.md                # Project documentation
├── main.py                 # Entry point to run the game or selected AI agent
├── app/
│   └── tetris_dual.py      # Core Tetris game engine
├── bots/                   # AI agents
│   ├── heuristic_bot.py        # Rule-based baseline bot
│   ├── cnn_bot.py              # CNN-based inference agent
│   ├── dqn_bot_CNN.py          # DQN agent using CNN architecture
│   ├──lstm_bot.py  # Experimental LSTM-based agent (not fully functional)
│   └── cnn_training/           # Training environment for DQN agents
│       ├── tetris_env_cnn.py
│       ├── tetris_play_cnn.py
│       └── tetris_train_cnn.py
├── datasets/               # Pre-collected gameplay data for training / evaluation
│   ├── tetris_dataset_v1.csv
│   ├── tetris_dataset_v2.csv
│   └── tetris_dataset_cnn_XX.csv
├── models/                 # Trained models and weights
│   ├── tetris_cnn.pth
│   ├── tetris_dqn_v1.pth
│   ├── tetris_dqn_v2.pth
│   └── tetris_actor_lstm.pth
├── notebooks/             # Interactive training experiments
│   ├── CNN.ipynb
│   └── LSTMactor.ipynb
└── tools/                 # Data generation utilities
    ├── dataset_generator.py
    └── dataset_generator_random.py
```
## 🛠 Installation
```
git clone https://github.com/a-cory-k/Tetris_AI.git
cd Tetris_AI
pip install -r requirements.txt
```
# ▶️ Usage
Run the main script:
```
python main.py
```
After starting, a **main menu** will appear. You can choose one of the following modes:

| Mode | Description | Notes |
|------|-------------|-------|
| 🎮 **Player vs Player (PVP)** | Compete against a friend on the same computer. | Both players control their own tetrominoes. |
| 🤖 **Player vs Bot (PVB)** | Play against an AI agent. | You can select which bot to compete against (CNN, DQN, or Heuristic). |
| ⚔️ **Bot vs Bot (BVB)** | Watch two AI agents play against each other. | Useful for testing AI strategies and comparing performance. |
| 🕹️ **Single Player (SP)** | Play alone or watch a single AI agent play. | Choose a bot to observe its gameplay, or play manually yourself. |


# A demonstration of one DQN bot game

![Мой-фильм-2-2](https://github.com/user-attachments/assets/11650220-14e3-4f30-ba52-f264782b056f)





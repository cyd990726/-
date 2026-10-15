"""Sarsa 的 PyTorch Q 网络。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import time

import gymnasium as gym
import torch as th
from torch import Tensor, nn


class QNetwork(nn.Module):
    """输入状态，输出每个离散动作对应的 Q 值。"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, states: Tensor) -> Tensor:
        """返回 Q(s, ·)，形状为 ``(batch_size, action_dim)``。"""
        return self.layers(states)


@dataclass
class TrainConfig:
    # 网络参数
    state_dim: int
    action_dim: int
    hidden_dim: int = 128

    # Sarsa 的训练参数
    gamma: float = 0.99
    lr: float = 1e-3
    max_steps: int = 500_000
    device: str = "mps" if th.backends.mps.is_available() else "cpu"

    # 学习过程中的探索策略
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01


class Sarsa:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.q_network = QNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(config.device)
        self.optimizer = th.optim.Adam(self.q_network.parameters(), lr=config.lr)

    def choose_action(self, state, epsilon: float) -> int:
        """根据 epsilon-greedy 策略选择动作。"""
        if random.random() < epsilon:
            return random.randint(0, self.config.action_dim - 1)

        state_tensor = th.tensor(
            state,
            dtype=th.float32,
            device=self.config.device,
        ).unsqueeze(0)
        with th.no_grad():
            return self.q_network(state_tensor).argmax(dim=1).item()

    def update(self, state, action, reward, next_state, next_action, terminated) -> float:
        """用一条 on-policy 的 S, A, R, S', A' 样本更新 Q 网络。"""
        state_tensor = th.tensor(state, dtype=th.float32, device=self.config.device).unsqueeze(0)
        next_state_tensor = th.tensor(next_state, dtype=th.float32, device=self.config.device).unsqueeze(0)
        action_tensor = th.tensor([[action]], dtype=th.int64, device=self.config.device)
        next_action_tensor = th.tensor([[next_action]], dtype=th.int64, device=self.config.device)
        reward_tensor = th.tensor([reward], dtype=th.float32, device=self.config.device)
        terminated_tensor = th.tensor([terminated], dtype=th.float32, device=self.config.device)

        curr_q_val = self.q_network(state_tensor).gather(1, action_tensor).squeeze(1)

        with th.no_grad():
            next_q_val = self.q_network(next_state_tensor).gather(1, next_action_tensor).squeeze(1)
            target_q_val = reward_tensor + self.config.gamma * next_q_val * (1 - terminated_tensor)

        loss = nn.functional.mse_loss(curr_q_val, target_q_val)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save(self, checkpoint_path: str | Path) -> None:
        """保存 Q 网络参数。"""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        th.save(self.q_network.state_dict(), checkpoint_path)

    def eval(
        self,
        checkpoint_path: str | Path,
        env_id: str,
        render_fps: int = 30,
    ) -> float:
        """加载模型，在指定环境中贪心执行并渲染一个完整回合。"""
        if render_fps <= 0:
            raise ValueError("render_fps 必须大于 0")

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"找不到模型参数文件：{checkpoint_path}")

        state_dict = th.load(
            checkpoint_path,
            map_location=self.config.device,
            weights_only=True,
        )
        self.q_network.load_state_dict(state_dict)
        self.q_network.eval()

        env = gym.make(env_id, render_mode="human")
        state, _ = env.reset()
        episode_reward = 0.0
        done = False

        try:
            env.render()
            while not done:
                frame_start = time.perf_counter()
                action = self.choose_action(state, epsilon=0.0)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward

                remaining = 1.0 / render_fps - (time.perf_counter() - frame_start)
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            env.close()

        print(f"Evaluation reward: {episode_reward:.0f}")
        return episode_reward

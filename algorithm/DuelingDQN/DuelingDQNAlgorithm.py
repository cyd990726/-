"""Dueling DQN 的 PyTorch Q 网络。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random
import time

import gymnasium as gym
import numpy as np
import torch as th
from torch import Tensor, nn


class DuelingQNetwork(nn.Module):
    """分别估计状态价值 V(s) 和动作优势 A(s, a)。"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # 共享的全链接层
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        # 状态价值头
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # 优势头
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    # 和DQN最大的差别就是网络结构和前向传播中算Q值部分了
    def forward(self, states: Tensor) -> Tensor:
        """返回 Q(s, a)=V(s)+A(s,a)-mean_a A(s,a)。"""
        features = self.feature_layer(states)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + advantages - advantages.mean(dim=1, keepdim=True)


class ReplayBuffer:
    # 经验回放缓冲区。
    def __init__(self, replay_buffer_size: int = 10000):
        self.replay_buffer_size = replay_buffer_size
        self.buffer = deque(maxlen=replay_buffer_size)

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int = 64):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones


@dataclass
class TrainConfig:
    # 网络参数
    state_dim: int
    action_dim: int
    hidden_dim: int = 128

    # Dueling DQN 的训练参数
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    replay_buffer_size: int = 500
    target_update_freq: int = 100
    max_steps: int = 500_000
    device: str = "mps" if th.backends.mps.is_available() else "cpu"

    # 学习过程中的探索策略
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01


class DuelingDQN:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.q_network = DuelingQNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(config.device)
        self.target_network = DuelingQNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(config.device)
        self.update_target_network()

        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)
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

    def update(self, cur_steps, checkpoint_inter, checkpoint_dir) -> float:
        """从回放池采样一批经验，并用 Double DQN 目标更新在线网络。"""
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)
        states = th.as_tensor(np.asarray(states), dtype=th.float32, device=self.config.device)
        actions = th.tensor(actions, dtype=th.int64, device=self.config.device)
        rewards = th.tensor(rewards, dtype=th.float32, device=self.config.device)
        next_states = th.as_tensor(np.asarray(next_states), dtype=th.float32, device=self.config.device)
        dones = th.tensor(dones, dtype=th.float32, device=self.config.device)

        curr_q_vals = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with th.no_grad():
            # Double DQN：在线网络选择 next action，目标网络评估这个 action。
            next_actions = self.q_network(next_states).argmax(dim=-1)
            next_q_vals = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_vals = rewards + self.config.gamma * next_q_vals * (1 - dones)

        loss = nn.functional.mse_loss(curr_q_vals, target_q_vals)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if cur_steps % checkpoint_inter == 0:
            checkpoint_path = checkpoint_dir / f"dueling_dqn_mountaincar_step_{cur_steps}.pt"
            self.save(checkpoint_path)
            print(f"Model saved to {checkpoint_path}, loss equal {loss}")

        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, checkpoint_path: str | Path) -> None:
        """保存在线 Q 网络参数。"""
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

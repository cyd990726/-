"""DQN 的 PyTorch Q 网络。"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time

import gymnasium as gym
import torch as th
from torch import Tensor, nn

from collections import deque
import random

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

class ReplayBuffer:
    # 经验回放缓冲区(默认10000条经验)
    def __init__(self, replay_buffer_size: int = 10000):
        self.replay_buffer_size = replay_buffer_size
        self.buffer = deque(maxlen=replay_buffer_size)

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # 一次默认采样64条经验
    def sample(self, batch_size:int = 64):
        # 随机生成采样索引
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            states,
            actions,
            rewards,
            next_states,
            dones
        )

@dataclass
class TrainConfig:
    # 网络参数
    state_dim: int 
    action_dim: int
    hidden_dim: int = 128

    # DQN 的训练参数
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    replay_buffer_size: int = 500
    target_update_freq: int = 100 # 目标更新频率
    max_steps :int = 50000
    device: str = "mps" if th.backends.mps.is_available() else "cpu"


    # 学习过程中的探索策略
    epsilon_start: float = 1.0 # 最开始基本是随机探索的
    epsilon_end: float = 0.01 #到后面逐渐倾向于我们学出来的策略，但是仍然保留一定的随机性




class DQN:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        # DQN中首先应该包含Q网络
        self.q_network = QNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(device=config.device)
        self.target_network = QNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(device=config.device)
        # 同步参数
        self.update_target_network()

        # 经验缓冲池
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)

        # 优化器
        self.optimizer = th.optim.Adam(self.q_network.parameters(), lr=config.lr)

    # 选择动作
    def choose_action(self, state, epsilon: float) -> int:
        """根据 epsilon-greedy 策略选择动作。"""
        if random.random() < epsilon:
            return random.randint(0, self.config.action_dim - 1)

        state = th.tensor(state, 
                          dtype=th.float32, 
                          device= self.config.device).unsqueeze(0)
        with th.no_grad():
            return self.q_network(state).argmax(dim=1).item()

    # 从回放池采样一批经验，并更新一次在线网络。
    def update(self):
        # 先从经验缓冲池采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)
        states = th.tensor(states, dtype=th.float32).to(self.config.device)
        actions = th.tensor(actions, dtype=th.int64).to(self.config.device)
        rewards = th.tensor(rewards, dtype=th.float32).to(self.config.device)
        next_states = th.tensor(next_states, dtype=th.float32).to(self.config.device)
        dones = th.tensor(dones, dtype=th.float32).to(self.config.device)
        # 计算当前 Q 值
        curr_q_vals = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
  
        with th.no_grad():
            next_q_vals = self.target_network(next_states).max(1)[0]
            target_q_vals = rewards + self.config.gamma * next_q_vals * (1 - dones)
        # 计算损失
        loss = nn.functional.mse_loss(curr_q_vals, target_q_vals)
        # 反向传播和优化
        self.optimizer.zero_grad() #清空上一轮留下来的梯度
        loss.backward() # 反向传播重新计算梯度
        self.optimizer.step() # 根据刚刚计算出的梯度更新网络参数

    # 更新网络参数
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, checkpoint_path: str | Path) -> None:
        """保存在线 Q 网络的参数。"""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        th.save(self.q_network.state_dict(), checkpoint_path)

    def eval(self, checkpoint_path: str | Path, render_fps: int = 30) -> float:
        """加载模型，贪心执行并渲染一个完整的 CartPole 回合。"""
        if render_fps <= 0:
            raise ValueError("render_fps 必须大于 0")

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"找不到模型参数文件：{checkpoint_path}")

        # map_location 保证 CPU/MPS 均能正确加载参数。
        state_dict = th.load(
            checkpoint_path,
            map_location=self.config.device,
            weights_only=True,
        )
        self.q_network.load_state_dict(state_dict)
        self.q_network.eval()

        # Python 3.12 环境已安装 pygame，直接使用 Gymnasium 原生窗口渲染。
        env = gym.make("CartPole-v1", render_mode="human")
        state, _ = env.reset()
        episode_reward = 0.0
        done = False

        try:
            env.render()
            while not done:
                frame_start = time.perf_counter()
                # 评估时不探索：始终选择 Q 值最大的动作。
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


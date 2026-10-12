"""加载训练好的 DQN 模型，渲染一个完整评估回合。"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym

from DQNAlgorithm import DQN, TrainConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one greedy DQN evaluation episode.")
    parser.add_argument("--model", type=Path, default=Path("checkpoints/dqn_cartpole.pt"))
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    env = gym.make("CartPole-v1")
    try:
        config = TrainConfig(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
        )
    finally:
        env.close()

    DQN(config).eval(args.model, render_fps=args.fps)


if __name__ == "__main__":
    main()

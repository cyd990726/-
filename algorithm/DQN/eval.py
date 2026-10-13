"""加载训练好的 DQN 模型，渲染一个完整评估回合。"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import torch as th

from DQNAlgorithm import DQN, TrainConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one greedy DQN evaluation episode.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("checkpoints/dqn_mountaincar_step_50000.pt"),
    )
    parser.add_argument("--env", default="MountainCar-v0", help="Must match the environment used to train the model.")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    env = gym.make(args.env)
    try:
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError("当前评估仅支持一维连续状态空间。")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("当前评估仅支持离散动作空间。")
        config = TrainConfig(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
        )

        # 在创建 DQN 前检查 checkpoint 与目标环境是否匹配，避免出现难懂的矩阵维度错误。
        checkpoint = th.load(args.model, map_location="cpu", weights_only=True)
        checkpoint_state_dim = checkpoint["layers.0.weight"].shape[1]
        checkpoint_action_dim = checkpoint["layers.4.weight"].shape[0]
        if (checkpoint_state_dim, checkpoint_action_dim) != (config.state_dim, config.action_dim):
            raise ValueError(
                f"模型与环境不匹配：checkpoint 要求 state_dim={checkpoint_state_dim}, "
                f"action_dim={checkpoint_action_dim}；{args.env} 提供 "
                f"state_dim={config.state_dim}, action_dim={config.action_dim}。"
            )
    finally:
        env.close()

    DQN(config).eval(args.model, env_id=args.env, render_fps=args.fps)


if __name__ == "__main__":
    main()

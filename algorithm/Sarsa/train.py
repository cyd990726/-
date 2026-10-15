from pathlib import Path

import gymnasium as gym

from SarsaAlgorithm import Sarsa, TrainConfig


# 模型保存间隔步数
CHECKPOINT_INTERVAL = 1_000
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

ENV_ID = "MountainCar-v0"
env = gym.make(ENV_ID)

config = TrainConfig(
    # 网络参数
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    max_steps=500_000,
)

agent = Sarsa(config=config)


def get_epsilon(total_steps: int) -> float:
    epsilon = (
        config.epsilon_start
        + (config.epsilon_end - config.epsilon_start) * total_steps / config.max_steps
    )
    return max(config.epsilon_end, epsilon)


def shape_mountaincar_reward(state, reward, next_state, terminated):
    """为 MountainCar 提供更密集的训练信号。"""
    previous_position = float(state[0])
    position = float(next_state[0])
    velocity = float(next_state[1])

    shaped_reward = reward
    shaped_reward += 10.0 * (position - previous_position)
    shaped_reward += 2.0 * (position + 0.5)
    shaped_reward += 10.0 * abs(velocity)

    if terminated:
        shaped_reward += 100.0

    return shaped_reward


episode_reward = 0.0
episode_env_reward = 0.0
episode_count = 0
total_steps = 0
latest_loss = 0.0

state, _ = env.reset()
action = agent.choose_action(state, get_epsilon(total_steps))

while total_steps < config.max_steps:
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    shaped_reward = shape_mountaincar_reward(state, reward, next_state, terminated)

    next_epsilon = get_epsilon(total_steps + 1)
    next_action = agent.choose_action(next_state, next_epsilon)
    latest_loss = agent.update(
        state=state,
        action=action,
        reward=shaped_reward,
        next_state=next_state,
        next_action=next_action,
        terminated=terminated,
    )

    state = next_state
    action = next_action
    episode_reward += shaped_reward
    episode_env_reward += reward
    total_steps += 1

    # 每 1,000 个环境交互步保存一次，文件名包含当前训练步数。
    if total_steps % CHECKPOINT_INTERVAL == 0:
        checkpoint_path = CHECKPOINT_DIR / f"sarsa_mountaincar_step_{total_steps}.pt"
        agent.save(checkpoint_path)
        print(f"Model saved to {checkpoint_path}, loss={latest_loss:.6f}")

    # 一个 episode 结束后，重置环境并按当前策略选择新 episode 的第一个动作。
    if done:
        episode_count += 1
        print(
            f"episode={episode_count}, "
            f"steps={total_steps}, "
            f"reward={episode_reward:.1f}, "
            f"env_reward={episode_env_reward:.0f}, "
            f"epsilon={get_epsilon(total_steps):.3f}, "
            f"loss={latest_loss:.6f}"
        )
        state, _ = env.reset()
        action = agent.choose_action(state, get_epsilon(total_steps))
        episode_reward = 0.0
        episode_env_reward = 0.0

env.close()

# 若训练在非 1,000 的整数倍停止，额外保存最后一次参数。
if total_steps % CHECKPOINT_INTERVAL != 0:
    checkpoint_path = CHECKPOINT_DIR / f"sarsa_mountaincar_step_{total_steps}.pt"
    agent.save(checkpoint_path)
    print(f"Final model saved to {checkpoint_path}")

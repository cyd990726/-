import gymnasium as gym
import torch as th
from pathlib import Path

from DQNAlgorithm import TrainConfig
from DQNAlgorithm import DQN
# 模型保存间隔步数
CHECKPOINT_INTERVAL = 1_000
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

ENV_ID = "MountainCar-v0"
env = gym.make(ENV_ID)

config = TrainConfig(
    # 网络参数
    state_dim = env.observation_space.shape[0],
    action_dim = env.action_space.n,
    max_steps = 2_000_000,
)

agent = DQN(config=config)

episode_reward = 0.0
episode_count = 0
total_steps = 0

state, _ = env.reset()

while total_steps < config.max_steps:
    epsilon = (
        config.epsilon_start
        + (config.epsilon_end - config.epsilon_start)*total_steps/config.max_steps
    )

    # 选择、执行动作
    action = agent.choose_action(state, epsilon)
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    # 将状态转移样本加入回放池
    agent.replay_buffer.add(state, action, reward, next_state, terminated)

    # 状态转移
    state = next_state
    episode_reward += reward
    total_steps += 1

    if len(agent.replay_buffer) >= config.batch_size:
        # 说明可以开始采样更新网络了
        agent.update(total_steps, CHECKPOINT_INTERVAL, CHECKPOINT_DIR)

    if total_steps % config.target_update_freq == 0:
        agent.update_target_network()

    # 每 1,000 个环境交互步保存一次，文件名包含当前训练步数。
    if total_steps % CHECKPOINT_INTERVAL == 0:
        checkpoint_path = CHECKPOINT_DIR / f"dqn_mountaincar_step_{total_steps}.pt"
        agent.save(checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

    # 一个episode结束了，要及时开启下一个episode
    if done:
        episode_count += 1
        print(
            f"episode={episode_count}, "
            f"steps={total_steps}, "
            f"reward={episode_reward:.0f}, "
            f"epsilon={epsilon:.3f}"
        )
        state, _ = env.reset()
        episode_reward = 0.0

env.close()

# 若训练在非 1,000 的整数倍停止，额外保存最后一次参数。
if total_steps % CHECKPOINT_INTERVAL != 0:
    checkpoint_path = CHECKPOINT_DIR / f"dqn_mountaincar_step_{total_steps}.pt"
    agent.save(checkpoint_path)
    print(f"Final model saved to {checkpoint_path}")



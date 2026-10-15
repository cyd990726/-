# PyTorch Sarsa（MountainCar-v0）

这是一个使用 PyTorch 实现的基础 Deep Sarsa。整体文件结构和 `algorithm/DQN` 保持一致，方便对比两个算法的差异。

训练时使用：

- ε-greedy 探索；
- 在线 Q 网络；
- Sarsa 的 on-policy TD 目标；
- MSE TD 损失；
- Adam 与反向传播；
- Apple Silicon 上自动使用 MPS；其他设备回退到 CPU。

Sarsa 不使用经验回放池，也不使用目标 Q 网络。每一步更新都来自当前策略实际走出的 `S, A, R, S', A'`。

环境使用维护中的 `gymnasium`，与项目当前的 NumPy 2 兼容；不再依赖已停止维护的 `gym`。

## 文件职责

- `SarsaAlgorithm.py`：`QNetwork`、`TrainConfig` 和 `Sarsa` 类。
- `train.py`：按环境交互步数训练，并定期保存 checkpoint。
- `eval.py`：加载 checkpoint，以贪心策略渲染一个完整评估回合。

## 训练

从本目录执行：

```bash
../../venv/bin/python train.py
```

训练默认执行 500,000 个环境交互步数。每训练 1,000 步，都会在本目录的 `checkpoints/` 下保存模型，例如 `sarsa_mountaincar_step_1000.pt`、`sarsa_mountaincar_step_2000.pt`。训练步数、网络规模、学习率等参数可在 `TrainConfig` 中调整。

MountainCar 的原始奖励每步都是 `-1`，对基础 Sarsa 来说信号比较稀疏。训练时会把用于更新的奖励改成塑形奖励：保留原始步惩罚，同时根据位置推进量、下一状态的位置、速度和到达终点的情况增加奖励。日志里的 `reward` 是塑形后的训练回报，`env_reward` 是环境原始回报。

## 渲染评估

选择一个训练好的 checkpoint，加载并渲染一个完整回合：

```bash
../../venv/bin/python eval.py --model checkpoints/sarsa_mountaincar_step_500000.pt
```

评估直接使用 Gymnasium 基于 pygame 的原生窗口。窗口默认以 30 FPS 播放；可通过 `--fps 60` 调整。评估采用贪心策略（不随机探索），回合结束后自动关闭窗口。

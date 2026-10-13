# PyTorch DQN（MountainCar-v0）

这是一个使用 PyTorch 实现的 DQN。网络继承 `torch.nn.Module`，训练时使用：

- ε-greedy 探索；
- 经验回放池；
- 在线 Q 网络和目标 Q 网络；
- MSE TD 损失；
- Adam 与反向传播；
- Apple Silicon 上自动使用 MPS；其他设备回退到 CPU。

环境使用维护中的 `gymnasium`，与项目当前的 NumPy 2 兼容；不再依赖已停止维护的 `gym`。

## 文件职责

- `DQNAlgorithm.py`：`QNetwork`、`ReplayBuffer`、`TrainConfig` 和 `DQN` 类。
- `train.py`：按环境交互步数训练，并在结束后保存 checkpoint。
- `eval.py`：加载 checkpoint，以贪心策略渲染一个完整评估回合。

## 训练

从本目录执行：

```bash
../../venv/bin/python train.py
```

训练默认执行 50,000 个环境交互步数。每训练 1,000 步，都会在本目录的 `checkpoints/` 下保存模型，例如 `dqn_mountaincar_step_1000.pt`、`dqn_mountaincar_step_2000.pt`。训练步数、回放池大小、目标网络同步频率等参数可在 `TrainConfig` 中调整。

## 渲染评估

选择一个训练好的 checkpoint，加载并渲染一个完整回合：

```bash
../../venv/bin/python eval.py --model checkpoints/dqn_mountaincar_step_50000.pt
```

评估直接使用 Gymnasium 基于 pygame 的原生窗口。窗口默认以 30 FPS 播放；可通过 `--fps 60` 调整。评估采用贪心策略（不随机探索），回合结束后自动关闭窗口。

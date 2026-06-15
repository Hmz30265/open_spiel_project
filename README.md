# 线上测试 v1 报告

> ⏱ 花费时间：5–7 小时

---

## 环境配置

本项目使用 **Python 3.11**，安装依赖：

```bash
pip install -r requirements.txt
```

---

## 任务 A

### A1：随机策略 Baseline

双方随机策略对打 ≥ 10000 局，统计先手/后手平均收益：

```bash
python open_spiel/python/examples/mcts.py \
  --game=kuhn_poker \
  --player1=random \
  --player2=random \
  --num_games=10000
```

**结果：**

```
Number of games played: 10000
Number of distinct games played: 30
Players: random random
Overall wins  [5667, 4333]
Overall returns [1394.0, -1394.0]
```

---

### A2：PPO 训练

`ppo_example.py` 未设置自我对弈策略，因此选择**对抗随机策略**。

**训练命令：**

```bash
# 清除旧日志（Windows）
rmdir /s /q runs

# 开始训练
python open_spiel/python/examples/ppo_example.py \
  --game_name=kuhn_poker \
  --num_envs=8 \
  --total_timesteps=100000 \
  --cuda
```

**查看训练曲线（TensorBoard）：**

```bash
tensorboard --logdir=runs
```

访问 [http://localhost:6006](http://localhost:6006)

**训练曲线：**

![Training Reward](training_reward.png)

| 指标 | 数值 |
|------|------|
| 对随机策略平均收益 | **0.417** |

---

### A3：Exploitability 对比

#### PPO 策略可利用度

```bash
python eval_explo.py
```

```
Exploitability: 0.416722
```

#### CFR 策略可利用度

```bash
python open_spiel/python/examples/cfr_example.py --game=kuhn_poker
```

| 迭代次数 | Exploitability |
|----------|----------------|
| 0        | 0.458333       |
| 10       | 0.060469       |
| 20       | 0.039914       |
| 30       | 0.024167       |
| 40       | 0.020517       |
| 50       | 0.014479       |
| 60       | 0.014004       |
| 70       | 0.011778       |
| 80       | 0.010102       |
| 90       | 0.009834       |

> CFR 经过 90 次迭代后可利用度降至 **0.0098**，远低于 PPO 的 **0.4167**。

---

## 任务 B

### B1：PPO 距离纳什均衡有多远？

当前 PPO 仅与随机策略对打，exploitability = **0.4167**，接近随机策略基准（~0.5），说明离纳什均衡还很远。

**为什么自我对弈 PPO 不保证收敛到纳什均衡？**

- PPO 本质上是**最大化自身 reward**，而非寻找均衡策略
- 自我对弈存在**循环问题**：Player 0 更新策略 → Player 1 随之改变 → Player 0 又需更新，形成无限循环
- 不完美信息博弈中，局部最优不等于全局均衡

---

### B2：密集奖励方案设计

将终局奖励改为每步密集奖励后：

- **优点**：对随机策略的学习能力可能略有提升
- **缺点**：追求每步最高奖励可能导致过度虚张声势（bluff），一旦对手识破策略，整体收益反而下降

**设计方案（基于最小化遗憾）：**

| 行为 | Reward |
|------|--------|
| 拿到大牌，弃牌 | -0.5 |
| 拿到小牌，弃牌 | -0.1 |
| 拿到小牌，下注（bluff）| -0.7 |
| 拿到大牌，下注 | +0.3 |

> 思路类似 CFR：保守稳健，稳步积累筹码，而非追求高风险高收益。

**可能被钻空子的方式：**

对手一旦发现我方永远弃掉小牌，便会主动 bluff。当双方都持小牌时，我方必输。对局拉长后，筹码将持续流失。

---

### B3：真实遇到的失败 / 异常

原本计划自己设计 PPO，发现 repo 中已有相关实现便直接借用。

**遇到的问题：**

运行时出现 `tensor shape not matching` 错误，部分地方误用了 `list` 而非 `tuple`。

**定位与解决：**

报错信息精准定位到具体行号，逐一修改类型即可。整体无重大 bug。

训练不收敛属于**预期内现象**——PPO 对抗随机策略本身就存在收敛性问题。

---

## AI 协作说明

本项目使用 **Claude** 辅助完成以下工作：

Repo 本身文档较为详细，执行命令时 AI 使用较少，仅在 flag 不确定时少量参考。
- ✅ 编写 `eval_explo.py`（计算 exploitability）
- ✅ Debug 及分析报错信息（中度使用）
- ✅ 验证任务 B 中的观点
- ✅ 整理 `README.md` 与 `requirements.txt`


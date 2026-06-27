  # Day 2

## 今日目标

1. 学会使用yaml文件+命令行传参
2. 学会看tensorboard图像并分析参数好坏

## 任务1

1. 数据流：
```bash
命令行输入 --config
         ↓
 parse_args()
         ↓
 得到配置文件路径
         ↓
 load_config(config_path)
         ↓
 读取 YAML，变成 Python dict
         ↓
 train(config)
         ↓
 从 config 里取参数
         ↓
 创建 Gymnasium 环境
         ↓
 创建 PPO 模型
         ↓
 model.learn(...) 训练
         ↓
 保存模型和 TensorBoard 日志
```
2. 程序启动方式
```bash
python train_tuning.py --config configs/confibaseline.yaml
```

## 任务2

1. tensorboard图像含义
```bash
- rollout/ep_rew_mean：最近一段采样里，episode 的平均奖励。上升早，说明学的快，平台高说明最终效果好，波动小，说明效果稳

- rollout/ep_len_mean：最近一段采样里，episode 平均长度。对 CartPole 来说，和 reward 几乎一样，所以你会看到两条曲线数值非常接近。

- time/fps：训练速度，每秒跑多少环境步。这个主要看性能，不看学习效果。

- train/approx_kl：新旧策略差异大小。越大说明每次更新改得越猛，太大通常不稳。

- train/clip_fraction：有多少样本触发了 PPO 的裁剪。越高说明更新越激进，越容易被 clip。

- train/clip_range：裁剪阈值。你这里固定是 0.2，所以它不是结果，只是超参数。

- train/entropy_loss：注意它通常是“负熵”，数值越接近 0，说明策略越确定、探索越少；更负则表示更随机。前期越随机通常更好，因为需要探索，后期稳定更好因为已经接近收敛

- train/explained_variance：价值函数拟合得好不好，critic预测和真实回报的差异。接近 1 说明 critic 很准，接近 0 说明没比瞎猜强多少，负数说明比简单的平均值还差

- train/learning_rate：学习率，本实验里是固定值。

- train/loss：总损失，混合了 policy/value/entropy 等项，不适合单独解读。

- train/policy_gradient_loss：策略更新部分的损失，通常看趋势，不看单点。

- train/value_loss：价值函数误差，通常越低越好，但前期波动很正常。
```
2. 重点参数

先看学没学会，再看训练稳不稳
```bash
rollout/ep_rew_mean       学没学会
train/approx_kl           是不是学太猛
train/clip_fractioncritic 是否跟上了
train/explained_variance  哪组参数更稳
```

   # PPO 超参数对比实验

  ## 实验设置
```bash
实验名           baseline
total_timesteps  50000
learning_rate    0.0003
gamma            0.99
n_steps          2048
batch_size       64
目的             作为对照组
───────────────────────────────────────────────────────────
实验名           lr_high
total_timesteps  50000
learning_rate    0.001
gamma            0.99
n_steps          2048
batch_size       64
目的             观察学习率变大是否更快/更不稳定
──────────────────────────────────────────────────────────
实验名           gamma_low
total_timesteps  50000
learning_rate    0.0003
gamma            0.95
n_steps          2048
batch_size       64
目的             观察降低长期奖励权重的影响
```
## 实验结果

1. 三组的rollout/ep_rew_mean全都稳步上升，但是没有趋于稳定，其中lr_high上升的最早学的最快，最终值最大效果最好，gamma_low次之，baseline最次

2. train/approx_kl前期lr_high最高，gamma_low次之，实验初期策略变化大，后期lr_high最低，gamma_low次之，说明经历了前期的激烈迭代，后续策略已经趋于稳定，而baseline全程波动不大，比较保守，lr_high、gamma_low更激进。train/clip_fraction的趋势与之相似

3. train/explained_variance，前期三组都接近1,r_high最高，gamma_low次之，baseline最次，后期baseline仍然接近1,r_high接近0,gamma_low出现较大负值










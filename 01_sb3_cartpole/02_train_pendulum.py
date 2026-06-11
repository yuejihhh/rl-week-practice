from pathlib import Path
import gymnasium as gym
from stable_baselines3 import PPO

# /home/zhangyue/rl-week-practice/01_sb3_cartpole
ROOT_DIR = Path(__file__).resolve().parent

# /home/zhangyue/rl-week-practice/01_sb3_cartpole/models
MODEL_DIR = ROOT_DIR / "models"

# /home/zhangyue/rl-week-practice/01_sb3_cartpole/logs
LOG_DIR = ROOT_DIR / "logs"
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


# 定义主函数 main()。
# 把主要逻辑放进 main() 是 Python 项目里的常见写法，代码结构更清楚。
def main():
    # 创建 Pendulum-v1 环境。
    # 在 Pendulum 中，环境负责维护摆锤的状态。
    # 智能体每一步会从 env 里拿到 observation，然后把 action 交给 env 执行。
    env = gym.make("Pendulum-v1")

    model = PPO(
        # Pendulum 的 observation 是 3 个数字，不是图片，所以用 MlpPolicy 合适。
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        verbose=1,
        tensorboard_log=str(LOG_DIR),
    )

    model.learn(
        
        total_timesteps=200_000,

        # tb_log_name 是 TensorBoard 里这次实验的名字。
        # 日志目录通常会类似 logs/ppo_pendulum_1/。
        tb_log_name="ppo_pendulum",
    )

    # 保存训练好的模型。
    # MODEL_DIR / "ppo_pendulum" 会得到 models/ppo_pendulum 这个路径。
    # Stable-Baselines3 保存时会自动生成 ppo_pendulum.zip。
    # 这个 zip 文件里包含训练好的神经网络参数和算法配置。
    model.save(MODEL_DIR / "ppo_pendulum")

    # 关闭环境，释放资源。
    # 对简单环境影响不大，但这是好习惯。
    # 如果环境有图形窗口、视频录制或仿真器，close() 更重要。
    env.close()

if __name__ == "__main__":
    # 调用 main()，真正开始执行上面定义的训练流程。
    main()

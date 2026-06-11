# 从 Python 标准库 pathlib 中导入 Path。
# Path 用来处理文件路径，比直接拼字符串更安全、清晰。
from pathlib import Path

# 导入 Gymnasium，并把它简写为 gym。
# Gymnasium 提供强化学习环境，例如 CartPole-v1。
import gymnasium as gym

# 从 Stable-Baselines3 导入 PPO 算法。
# PPO 是这里用来训练智能体的强化学习算法。
from stable_baselines3 import PPO


# __file__ 表示当前这个 Python 文件的路径。
# Path(__file__) 把文件路径转换成 Path 对象。
# resolve() 把路径转换成绝对路径。
# parent 取当前文件所在的父目录。
# 如果当前文件是:
# /home/zhangyue/rl-week-practice/01_sb3_cartpole/train_cartpole.py
# 那么 ROOT_DIR 就是:
# /home/zhangyue/rl-week-practice/01_sb3_cartpole
ROOT_DIR = Path(__file__).resolve().parent

# 定义保存模型的目录。
# / 在 Path 对象里表示“拼接路径”，不是数学除法。
# 最终路径大概是:
# /home/zhangyue/rl-week-practice/01_sb3_cartpole/models
MODEL_DIR = ROOT_DIR / "models"

# 定义保存训练日志的目录。
# TensorBoard 会读取这个目录里的日志文件，然后显示训练曲线。
# 最终路径大概是:
# /home/zhangyue/rl-week-practice/01_sb3_cartpole/logs
LOG_DIR = ROOT_DIR / "logs"

# 创建 models/ 目录。
# exist_ok=True 的意思是: 如果目录已经存在，就不要报错。
# 这样脚本可以重复运行。
MODEL_DIR.mkdir(exist_ok=True)

# 创建 logs/ 目录。
# 如果没有这个目录，后面保存 TensorBoard 日志时可能会失败。
LOG_DIR.mkdir(exist_ok=True)


# 定义主函数 main()。
# 把主要逻辑放进 main() 是 Python 项目里的常见写法，代码结构更清楚。
def main():
    # 创建 CartPole-v1 环境。
    # env 是 environment 的缩写，可以理解为“游戏世界”。
    # 在 CartPole 中，环境负责维护小车和杆子的状态。
    # 智能体每一步会从 env 里拿到 observation，然后把 action 交给 env 执行。
    env = gym.make("CartPole-v1")

    # 创建 PPO 模型，也就是创建一个准备训练的智能体。
    # 注意: 这一行只是创建模型，模型一开始还没有学会控制小车。
    # 训练真正发生在后面的 model.learn(...)。
    model = PPO(
        # policy 指定策略网络类型。
        # MlpPolicy 表示使用多层全连接神经网络。
        # CartPole 的 observation 是 4 个数字，不是图片，所以用 MlpPolicy 合适。
        policy="MlpPolicy",

        # env 指定训练环境。
        # PPO 会不断和这个 CartPole 环境交互，收集经验，再用经验更新策略。
        env=env,

        # learning_rate 是学习率。
        # 3e-4 是科学计数法，等于 0.0003。
        # 它控制神经网络每次更新参数时“迈多大步”。
        # 太大可能训练不稳定，太小可能学得很慢。
        learning_rate=3e-4,

        # n_steps 表示 PPO 每次更新模型前，先收集多少步环境交互数据。
        # *这里是 2048 步。指的是单个环境交互的步数，不是总训练步数。
        # 每一步通常包含: observation、action、reward、下一步 observation、是否结束。
        n_steps=2048,

        # batch_size 表示训练神经网络时，每个小批次使用多少条数据。
        # PPO 会先收集 n_steps 条数据，然后拆成多个小批次做训练。
        # 这里每次取 64 条经验更新一次网络参数。
        batch_size=64,

        # gamma 是折扣因子，用来控制智能体有多重视未来奖励。
        # 0.99 很接近 1，表示比较重视长期收益。
        # *CartPole 的目标是让杆子尽可能久不倒，所以未来奖励很重要。
        gamma=0.99,

        # verbose 控制训练时终端输出多少日志。
        # 0 表示基本不输出，1 表示输出训练过程中的关键信息。
        verbose=1,

        # tensorboard_log 指定 TensorBoard 日志目录。
        # LOG_DIR 是 Path 对象，str(LOG_DIR) 把它转换成字符串路径。
        # 训练后可以用 tensorboard 查看 reward、loss 等曲线。
        tensorboard_log=str(LOG_DIR),
    )

    # 开始训练 PPO 模型。
    # learn(...) 会让模型反复执行: 观察环境 -> 选择动作 -> 获得奖励 -> 更新策略。
    model.learn(
        # total_timesteps 是总训练步数。
        # 50_000 和 50000 完全一样，下划线只是让数字更好读。
        # 这里表示智能体总共和环境交互 50000 步。
        total_timesteps=50_000,

        # tb_log_name 是 TensorBoard 里这次实验的名字。
        # 日志目录通常会类似 logs/ppo_cartpole_1/。
        tb_log_name="ppo_cartpole",
    )

    # 保存训练好的模型。
    # MODEL_DIR / "ppo_cartpole" 会得到 models/ppo_cartpole 这个路径。
    # Stable-Baselines3 保存时会自动生成 ppo_cartpole.zip。
    # 这个 zip 文件里包含训练好的神经网络参数和算法配置。
    model.save(MODEL_DIR / "ppo_cartpole")

    # 关闭环境，释放资源。
    # 对简单环境影响不大，但这是好习惯。
    # 如果环境有图形窗口、视频录制或仿真器，close() 更重要。
    env.close()


# 这是 Python 的常见入口写法。
# 当你直接运行 python train_cartpole.py 时，__name__ 会等于 "__main__"。
# 这时下面的 main() 会被调用，训练流程开始执行。
# 如果这个文件被别的 Python 文件 import，main() 不会自动运行。
if __name__ == "__main__":
    # 调用 main()，真正开始执行上面定义的训练流程。
    main()

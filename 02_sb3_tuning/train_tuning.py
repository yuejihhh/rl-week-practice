from pathlib import Path
#命令行参数解析库，用来接收 --config 参数。python train_tuning.py --config configs/config_baseline.yaml
import argparse

import gymnasium as gym
#用来读取 yaml 配置文件。把.yaml 文件里的内容加载成 Python 字典。
# learning_rate: 0.0003
# 读进 Python 后可以变成：
# {"learning_rate": 0.0003}
import yaml
from stable_baselines3 import PPO


ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"

MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


# 用来读取 yaml 配置文件并返回一个字典。
#config_path表示一个配置文件路径： configs/config_baseline.yaml
def load_config(config_path):
    # 读取 yaml 文件，返回一个 dict
    # open(...)：打开文件。
    # config_path：要打开的文件路径。
    # "r"：表示 read，只读模式。
    # encoding="utf-8"：表示用 UTF-8 编码读取文件，适合中文和常见文本。
    # as f：把打开的文件对象命名为 f。
    # with 的作用是：文件用完后自动关闭
    with open(config_path, "r", encoding="utf-8") as f:
        # yaml.safe_load(f) 读取 YAML 文件内容，并把它转换成 Python 字典。
        config = yaml.safe_load(f)
    return config




def train(config):
    # 1. 从 config 里取 experiment_name、env_id、total_timesteps、seed
    # 2. 从 config["ppo"] 里取 learning_rate、gamma、n_steps、batch_size
    # 3. 创建环境
    env = gym.make(config["env_id"])
    # 4. 创建 PPO 模型
    model = PPO(
        policy="MlpPolicy",
        env=env,
        seed=config["seed"],
        learning_rate=config["ppo"]["learning_rate"],
        n_steps=config["ppo"]["n_steps"],
        batch_size=config["ppo"]["batch_size"],
        gamma=config["ppo"]["gamma"],
        verbose=1,
        tensorboard_log=str(LOG_DIR),
    )
    # 5. 调用 model.learn(...)
    model.learn(
        total_timesteps=config["total_timesteps"],
        tb_log_name=config["experiment_name"]
        )
    # 6. 保存模型
    model.save(MODEL_DIR / config["experiment_name"])
    # 7. 关闭环境
    env.close()

# 定义 parse_args()，用 argparse 接收命令行 --config 参数得到配置文件路径。
def parse_args():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    #添加一个命令行参数：--config
    # required=True 表示这个参数必须提供。
    parser.add_argument("--config", required=True)
    # 解析命令行参数并返回一个包含参数的对象。
    return parser.parse_args()


def main():
    # 1. 解析命令行参数
    args = parse_args()
    # 2. 读取配置
    config=load_config(args.config)
    # 3. 调用 train(config)
    train(config)


if __name__ == "__main__":
    main()
    # 数据流
    # 命令行输入 --config
    #         ↓
    # parse_args()
    #         ↓
    # 得到配置文件路径
    #         ↓
    # load_config(config_path)
    #         ↓
    # 读取 YAML，变成 Python dict
    #         ↓
    # train(config)
    #         ↓
    # 从 config 里取参数
    #         ↓
    # 创建 Gymnasium 环境
    #         ↓
    # 创建 PPO 模型
    #         ↓
    # model.learn(...) 训练
    #         ↓
    # 保存模型和 TensorBoard 日志


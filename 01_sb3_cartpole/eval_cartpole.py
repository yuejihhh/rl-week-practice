# 从 Python 标准库 pathlib 中导入 Path。
# Path 用来处理文件路径，方便定位模型文件。
from pathlib import Path

# 导入 Gymnasium，并简写为 gym。
# 评估时仍然需要创建 CartPole-v1 环境，因为模型要在环境里运行。
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# 从 Stable-Baselines3 导入 PPO。
# 这里使用 PPO.load(...) 加载训练好的模型，而不是重新训练。
from stable_baselines3 import PPO


# 获取当前 eval_cartpole.py 文件所在目录。
# 如果当前文件路径是:
# /home/zhangyue/rl-week-practice/01_sb3_cartpole/eval_cartpole.py
# 那么 ROOT_DIR 就是:
# /home/zhangyue/rl-week-practice/01_sb3_cartpole
ROOT_DIR = Path(__file__).resolve().parent

# 指定训练脚本保存出来的模型文件。
# 训练代码 model.save(MODEL_DIR / "ppo_cartpole") 会生成 ppo_cartpole.zip。
# 所以评估代码要从 models/ppo_cartpole.zip 加载模型。
MODEL_PATH = ROOT_DIR / "models" / "ppo_cartpole.zip"
VIDEO_DIR = ROOT_DIR / "videos"


VIDEO_DIR.mkdir(exist_ok=True)  # 确保 videos 目录存在，如果已经存在就不报错。

# 定义主函数 main()。
# 这个函数负责加载模型、运行 10 局 CartPole、统计每局 reward。
def main():
    # 创建 CartPole-v1 环境。
    # 注意: 评估时也要创建环境，因为模型需要不断接收 observation 并执行 action。
    env = gym.make("CartPole-v1" , render_mode="rgb_array") # *使用RecordVideo需要添加render_mode="rgb_array" 
    env = RecordVideo(
        env, 
        video_folder=str(VIDEO_DIR), # 视频文件会保存在 videos 目录下
        episode_trigger=lambda episode_id: True, # 记录所有 episode 的视频
        name_prefix="ppo_cartpole" # 生成的视频文件会以 ppo_cartpole_episode_0.mp4、ppo_cartpole_episode_1.mp4 等命名
        )
    #*此时出现两个env，后一个环境并不会把前一个覆盖，而是对前一个环境进行包装，可以理解成：
    #* base_env = gym.make("CartPole-v1", render_mode="rgb_array")

    #* env = RecordVideo(
    #* base_env,
    #* video_folder=str(VIDEO_DIR),
    #* episode_trigger=lambda episode_id: True,
    #* name_prefix="ppo_cartpole",
    #* )

    

    # 加载已经训练好的 PPO 模型。
    # MODEL_PATH 指向 models/ppo_cartpole.zip。
    # 如果你还没有先运行 train_cartpole.py，这里会因为找不到模型文件而报错。
    model = PPO.load(MODEL_PATH)

    # 创建一个空列表，用来保存每一局的总 reward。
    # 例如评估 10 局后，它可能变成 [500, 500, 498, ...]。
    episode_rewards = []

    # for 循环用来重复评估多局。
    # range(10) 会生成 0,1,2,3,4,5,6,7,8,9，一共 10 个数字。
    # 所以这段代码会测试 10 个 episode，也就是 10 局游戏。
    for episode in range(10):
        # reset() 表示重置环境，开始新的一局。
        # Gymnasium 新版 reset() 返回两个值: obs 和 info。
        # obs 是 observation，也就是模型做决策需要看的环境状态。
        # CartPole 的 obs 有 4 个数字: 小车位置、小车速度、杆子角度、杆子角速度。
        # info 是环境额外信息，这里暂时用不到。
        obs, info = env.reset()

        # terminated 表示这一局是否因为任务本身结束。
        # 在 CartPole 里，常见原因是杆子倒得太厉害，或者小车跑出边界。
        # 一局刚开始时当然还没结束，所以设为 False。
        terminated = False

        # truncated 表示这一局是否因为时间限制结束。
        # CartPole-v1 最长通常是 500 步，如果坚持到时间上限，会 truncated=True。
        # 一局刚开始时还没有到时间上限，所以设为 False。
        truncated = False

        # total_reward 记录当前这一局累计拿到的奖励。
        # CartPole 默认每坚持一步 reward 加 1，所以总 reward 越高，说明坚持越久。
        total_reward = 0.0

        # while 循环表示: 只要这一局还没有结束，就一直让模型控制小车。
        # terminated or truncated 表示“任务结束或时间到”。
        # not (...) 表示“还没结束”。
        while not (terminated or truncated):
            # model.predict(...) 根据当前 observation 选择下一步 action。
            # CartPole 的 action 只有两个: 0 表示向左推，1 表示向右推。
            # deterministic=True 表示评估时选择模型认为最好的动作，而不是随机采样动作。
            # _states 是某些循环神经网络策略会用到的隐藏状态；这里的 MlpPolicy 用不到。
            action, _states = model.predict(obs, deterministic=True)

            # env.step(action) 把模型选择的 action 交给环境执行一步。
            # 执行后，环境会返回新的 obs、这一帧 reward、是否结束等信息。
            # obs: 执行动作后的新 observation。
            # reward: 这一步得到的奖励。CartPole 默认每多坚持一步就是 1。
            # terminated: 是否因为杆子倒了/小车出界等原因结束。
            # truncated: 是否因为达到最大步数等时间限制结束。
            # info: 额外调试信息，这里暂时不用。
            obs, reward, terminated, truncated, info = env.step(action)

            # 把当前这一步的 reward 累加到本局总分里。
            # 写法 total_reward += reward 等价于 total_reward = total_reward + reward。
            total_reward += reward

        # while 循环结束，说明这一局已经结束。
        # 把这一局的总 reward 保存到 episode_rewards 列表里，方便最后计算平均分。
        episode_rewards.append(total_reward)

        # 打印当前这一局的结果。
        # episode 从 0 开始，所以显示时用 episode + 1，让人看到 Episode 1 到 Episode 10。
        # f"..." 是 Python 的 f-string，可以把变量值放进字符串。
        print(f"Episode {episode + 1}: reward = {total_reward}")

    # 10 局全部结束后，计算平均 reward。
    # sum(episode_rewards) 是 10 局总分之和。
    # len(episode_rewards) 是列表长度，也就是评估了多少局，这里是 10。
    avg_reward = sum(episode_rewards) / len(episode_rewards)

    # 打印平均 reward。
    # CartPole-v1 单局满分通常是 500。
    # 如果平均 reward 接近 500，说明模型基本学会了保持杆子平衡。
    print(f"Average reward over {len(episode_rewards)} episodes: {avg_reward}")

    # 关闭环境，释放资源。
    # 即使没有打开图形窗口，结束时 close() 也是好习惯。
    env.close()


# 这是 Python 的常见入口写法。
# 当你直接运行 python eval_cartpole.py 时，__name__ 会等于 "__main__"，于是执行 main()。
# 如果这个文件被其他文件 import，main() 不会自动执行。
if __name__ == "__main__":
    # 调用 main()，开始执行评估流程。
    main()

from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

# /home/zhangyue/rl-week-practice/01_sb3_cartpole
ROOT_DIR = Path(__file__).resolve().parent

# 指定训练脚本保存出来的模型文件。
# 训练代码 model.save(MODEL_DIR / "ppo_pendulum") 会生成 ppo_pendulum.zip。
# 所以评估代码要从 models/ppo_pendulum.zip 加载模型。
MODEL_PATH = ROOT_DIR / "models" / "ppo_pendulum.zip"
VIDEO_DIR = ROOT_DIR / "videos"


VIDEO_DIR.mkdir(exist_ok=True)  # 确保 videos 目录存在，如果已经存在就不报错。

def main():
   
    env = gym.make("Pendulum-v1", render_mode="rgb_array") # *使用RecordVideo需要添加render_mode="rgb_array" 
    env = RecordVideo(
        env, 
        video_folder=str(VIDEO_DIR), # 视频文件会保存在 videos 目录下
        episode_trigger=lambda episode_id: True, # 记录所有 episode 的视频
        name_prefix="ppo_pendulum" # 生成的视频文件会以 ppo_pendulum_episode_0.mp4、ppo_pendulum_episode_1.mp4 等命名
        )
    model = PPO.load(MODEL_PATH)
    episode_rewards = []
    for episode in range(10):
  
        # Pendulum 的 obs 有 3 个数字: 角度、角速度、力矩。
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
           
            action, _states = model.predict(obs, deterministic=True)
            #*此处不需要修改，因为这是gym官方接口，任何环境都具有相同的接口规范，都会返回 obs, reward, terminated, truncated, info 这五个值。
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        episode_rewards.append(total_reward)

        print(f"Episode {episode + 1}: reward = {total_reward}")

    avg_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"Average reward over {len(episode_rewards)} episodes: {avg_reward}")

    env.close()

if __name__ == "__main__":
    main()

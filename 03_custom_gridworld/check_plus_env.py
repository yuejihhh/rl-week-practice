from stable_baselines3.common.env_checker import check_env

from gridworld_plus_env import GridWorldEnv


env = GridWorldEnv()

check_env(env)
print("check_env passed")

obs, info = env.reset()
print("reset:", obs, info)

terminated = False
truncated = False

total_reward = 0

while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    total_reward += reward

    print(
        "action=", action,
        "obs=", obs,
        "info=", info,
        "reward=", reward,
        "terminated=", terminated,
        "truncated=", truncated,
        "total_reward=", total_reward
    )

env.close()

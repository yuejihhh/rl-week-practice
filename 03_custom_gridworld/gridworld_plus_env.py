import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    #在环境对象创建时，把这个环境需要长期保存的重要属性先准备好。
    def __init__(self, size=5, max_steps=50):
        # *调用父类的构造函数，确保父类的初始化逻辑也被执行
        super().__init__()
        self.size = size
        self.max_steps = max_steps
        #spaces.Discrete(4) 表示一个“离散空间”，也就是只能从固定的几个整数选项里取一个值。
        #动作只能是下面 4 个整数之一：0, 1, 2, 3
        self.action_space = spaces.Discrete(4)
        #*BOX很适合表示连续向量空间
        #环境每次返回的 observation 是一个长度为 2 的数组，并且每个元素都在 [0, size - 1]之间。
        #是一个二维矩形范围
        #也可以用：spaces.MultiDiscrete([size, size])  区别是 MultiDiscrete 表示每个维度是离散的，而 Box 表示连续的范围。
        self.observation_space = spaces.Box(
            low=0,
            high=size - 1,
            shape=(2,),
            dtype=np.int32,
        )
        #Python 里变量不需要提前声明类型。不是在声明 agent_pos 永远是 None 类型，而是先占个位，表示“现在还没有智能体位置”
        self.agent_pos = None
        self.target_pos = None
        self.step_count = 0
        self.obstacles = []  # 障碍物位置列表，初始化为空

    def _get_obs(self):
        #创建一个 numpy 数组
        #*numpy数组可以进行向量化运算
        #普通 list：[1, 2] + [3, 4] 结果是拼接：[1, 2, 3, 4]
        #NumPy array：np.array([1, 2]) + np.array([3, 4]) 结果是逐元素相加：array([4, 6])
        return np.array(self.agent_pos, dtype=np.int32)

    def _get_info(self):
        distance = abs(self.agent_pos[0] - self.target_pos[0]) + abs(self.agent_pos[1] - self.target_pos[1])

        return {
            "distance": distance,
            "step_count": self.step_count,
            "agent_pos": self.agent_pos,
            "target_pos": self.target_pos,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        #直接给变量赋值为长度为2的列表，表示智能体和目标的位置
        self.agent_pos = [0, 0]
        self.target_pos = [self.size - 1, self.size - 1]
        self.step_count = 0
        self.obstacles = [[1,1],[2,1],[3,3]]  # 重置障碍物列表
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        self.step_count += 1
        last_pos = self.agent_pos.copy()  # 记录上一步的位置，所以要放在移动前，用于判断是否撞墙
        # 这里先放移动逻辑
        # 这里先放 reward / terminated / truncated 的判断
        if action == 0:  # up
            self.agent_pos[0] -= 1
        elif action == 1:  # down
            self.agent_pos[0] += 1
        elif action == 2:  # left
            self.agent_pos[1] -= 1
        elif action == 3:  # right
            self.agent_pos[1] += 1
        # 边界检查
        self.agent_pos[0] = int(np.clip(self.agent_pos[0], 0, self.size - 1))
        self.agent_pos[1] = int(np.clip(self.agent_pos[1], 0, self.size - 1))

        # *判断是否撞障碍物
        hit_obstacle = self.agent_pos in self.obstacles
        if hit_obstacle:
            self.agent_pos = last_pos  # 撞到障碍物，回到上一步的位置

        observation = self._get_obs()

        # *判断是否撞墙
        hit_wall = self.agent_pos == last_pos and not hit_obstacle  # 如果位置没变且不是撞障碍物，就说明撞墙了

        if self.agent_pos == self.target_pos:
            reward = 1
            terminated = True
        elif hit_wall:
            #如果智能体没有移动，说明撞墙了，给一个负奖励
            reward = -0.05
            terminated = False
        elif hit_obstacle:
            #如果智能体撞到障碍物，给一个负奖励
            reward = -0.05
            terminated = False
        else:
            #走的越久奖励越低，鼓励快速到达目标
            reward = -0.01
            terminated = False

        if self.step_count >= self.max_steps and not terminated:
            truncated = True
        else:
            truncated = False
        
        info = self._get_info()
        #*在 info 里添加一个新的键值对，表示是否撞墙
        info["hit_wall"] = hit_wall
        #*在 info 里添加一个新的键值对，表示是否撞到障碍物
        info["hit_obstacle"] = hit_obstacle

        return observation, reward, terminated, truncated, info

    def render(self):
        #print(f"agent={self.agent_pos}, target={self.target_pos},step={self.step_count}")
        for i in range(self.size):
            #为当前这一行创建一个空字符串，后续添加内容进去
            line = ""
            for j in range(self.size):
                if self.agent_pos == [i, j]:
                    line += "A "
                elif [i, j] in self.obstacles:
                    line += "# "
                elif self.target_pos == [i, j]:
                    line += "T "
                else:
                    line += ". "
            print(line)
        print("step=", self.step_count)

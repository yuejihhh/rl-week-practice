# Day 3 : 自定义 GridWorld 环境

## 今日目标

1. 学会创建girdworld环境

## 任务：

1. girdworld_env.py:创建一个GridWorldEnv类，继承gym.env

- __init__

  在环境创建时，把需要长期保存的重要属性（参数）先准备好

- _get_obs
  
  把内部变量返回为外部变量

- _get_info

  返回调试信息，因为info和obs是大部分函数的返回值，所以有这两个函数

- reset
 
  输入：seed、options

  输出：obs, info

  作用：重置一局，设置起始位置、终止位置，把物体重置到起始位置，并输出环境、观测信息

- step

  输入：action

  输出：obs, reward, terminated, truncated, info

  作用：根据动作更新状态、奖励；判断是否撞击障碍物；一局是否结束

- render

  渲染画面

2. check_env.py:检查环境是否符合gym规则并在环境中执行动作

  检查是否符合规则
  ```bash
  check_env(env)
  print("check_env passed")
  ```

3. 项目1：无障碍、起终点固定、稀疏奖励
- 网格大小：5x5
- 起点：左上角 (0, 0)
- 目标点：右下角 (4, 4)
- 障碍物：无
- 动作：4 个离散动作，上下左右
- 奖励：
    - 到达目标：+1
    - 其他步：-0.01
- 结束条件：
    - 到达目标：terminated=True
    - 超过最大步数：truncated=True
    - render输出：网格

4. 项目2：固定障碍物、区分撞墙和正常移动、

- 撞墙检测逻辑是：
```bash
    1. 移动前保存 last_pos
    2. 根据 action 更新 agent_pos
    3. 使用 np.clip 限制坐标不超出地图
    4. 如果移动后位置和 last_pos 相同，说明撞墙
```
- 奖励设计：
```bash
    到达目标：+1
    正常移动：-0.01
    撞墙：-0.05
```
- 并在 info 中加入：
```bash
    hit_wall=True/False
```

  这样可以观察每一步是否撞墙。

- 固定障碍物规则：
```bash
    agent 不能进入障碍物格子
    如果试图进入障碍物，则回到上一步位置
    撞障碍物会得到额外负奖励
```

- 奖励设计：
    
```bash
    到达目标：+1
    正常移动：-0.01
    撞墙：-0.05
    撞障碍物：-0.05
```
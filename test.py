import gym
import numpy as np

# 创建 Ant-v2 环境
env = gym.make("Ant-v2")

# 采集多个 state
states = []

obs = env.reset()

for _ in range(10000):
    action = env.action_space.sample()

    obs, reward, done, info = env.step(action)
    states.append(obs)

    if done:
        obs = env.reset()

states = np.array(states)

print("states shape:", states.shape)

# 计算每个维度的统计量
state_min = states.min(axis=0)
state_max = states.max(axis=0)
state_std = states.std(axis=0)

# 找出标准差为 0 的维度
zero_std_dims = np.where(state_std == 0)[0]

print("\n恒定维度:")
print(zero_std_dims)

print("\n恒定维度数量:")
print(len(zero_std_dims))

print("\n这些维度的值:")
print(states[0, zero_std_dims])

# 输出所有维度的统计信息
print("\n每个维度的统计信息:")
for i in range(states.shape[1]):
    print(
        f"dim {i:3d}: "
        f"min={state_min[i]:10.6f}, "
        f"max={state_max[i]:10.6f}, "
        f"std={state_std[i]:10.6f}"
    )

env.close()
import gymnasium as gym
import ale_py
import time

gym.register_envs(ale_py)
env = gym.make('ALE/Breakout-v5', render_mode='human')
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.render()

for i in range(1000):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

    time.sleep(0.01)

env.close()
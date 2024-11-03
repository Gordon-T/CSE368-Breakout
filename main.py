import gymnasium as gym
import ale_py
import time
import numpy as np

gym.register_envs(ale_py)

# Using "ram" obs space for now until we figure out how to make use of "rgb"
env = gym.make('ALE/Breakout-v5', render_mode='human', obs_type="ram")#, frameskip=1)
obs, info = env.reset()
env.render()

# TODO: Takes the rgb or grayscale observation space and tries to find where the ball and paddle x-coordinates
def findCoordinates(obs):
    # Matrix shape of obs should be (210, 160, 3)

    # Convert to grayscale by collapsing rgb channels into one
    # TODO: Might be possible to not grayscale the observation space?
    # - Problem is that both paddle, ball, and topmost layer of bricks shares rgb value of (200, 72, 72)
    grayscale = np.mean(obs, axis=2) 

    # Find where the pixels are brightest
    interested = np.where(grayscale > 50) # TODO: Have to look into this more, k-clustering?

    return ball_x, paddle_x

# Checks if we should move the paddle to meet the ball depending on the coordinates of each
def getAction(ball_x, paddle_x):
    if ball_x > paddle_x:
        return 2 # Move paddle right
    elif ball_x < paddle_x:
        return 3 # Move paddle left
    else:
        return 0 # Do nothing

# Fires the first ball
obs, reward, terminated, truncated, info = env.step(1)

terminated = False
# Agent loop
while not terminated:
    # Find ball and paddle positions
    #ball_x, paddle_x = findCoordinates(obs)

    #Fire ball
    env.step(1)
    
    ball_x = obs[99]
    paddle_x = obs[72]
    
    # Get the appropriate action
    action = getAction(ball_x, paddle_x)

    # Perform action
    #action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
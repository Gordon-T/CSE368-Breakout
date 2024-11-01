import gymnasium as gym
import ale_py
import numpy as np

gym.register_envs(ale_py)
env = gym.make('ALE/Breakout-v5', render_mode='human')
obs, info = env.reset()
env.render()

# TODO: Takes the observation space and tries to find where the ball and paddle x-coordinates
def findCoordinates(obs):
    # Matrix shape of obs should be (210, 160, 3)

    # Convert to grayscale by collapsing rgb channels into one
    # TODO: Might be possible to not grayscale the observation space?
    # - Problem is that both paddle, ball, and topmost layer of bricks shares rgb value of (200, 72, 72)
    grayscale = np.mean(obs, axis=2) 

    # Find where the pixels are brightest
    interested = np.where(grayscale > 50) # TODO: Have to look into this more, k-clustering?

    return ball_x, paddle_x

# Checks if we should move the paddle to meet the ball
def getAction(ball_x, paddle_x):
    if ball_x > paddle_x:
        return 2 # Move paddle right
    elif ball_x < paddle_x:
        return 3 # Move paddle left
    else:
        return 0 # Do nothing

# Fire the ball
env.step(1)

terminated = False
screen_width = obs.shape[1]

# Agent loop
while not terminated:
    # Find ball and paddle positions
    ball_x, paddle_x = findCoordinates(obs)
    
    # Get the appropriate action
    action = getAction(ball_x, paddle_x)
    
    # Perfom action
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()
        # Fire the ball again if we lose a life
        env.step(1)

env.close()
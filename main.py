import gymnasium as gym
import ale_py
import time
import numpy as np

gym.register_envs(ale_py)

# Using "ram" obs space for now until we figure out how to make use of "rgb"
env = gym.make('ALE/Breakout-v5', render_mode='human', obs_type="grayscale")#, frameskip=1)
obs, info = env.reset()
env.render()



# TODO: Takes the rgb or grayscale observation space and tries to find where the ball and paddle x-coordinates
def findCoordinates(obs):
    # observation was set to grayscale when we made the environment
    gray_frame = obs  #2D grayscale array

    bwFrame = gray_frame
    # make it so every pixel is either black (0) or white (255)

    for i in range(gray_frame.shape[0]):  # iterate over rows
        for j in range(gray_frame.shape[1]):  # iterate over columns
            if gray_frame[i, j] > 100:
                bwFrame[i, j] = 255  # set to white
            else:
                bwFrame[i, j] = 0  # set to black


    # Initialize positions as None to track ball and paddle locations
    ball_x = None
    paddle_x = None

    # Detect ball position by searching the upper part of the screen
    for y in range(100, 188):  # Limit search to the middle (between blocks and paddle)
        for x in range(10, 150):  # Scan the middle
            if bwFrame[y, x] == 255:  # Looking for white pixels (255) that represent the ball
                ball_x = x + 1 # Found the ball, track its x position
                break  # Break out of the inner loop once the ball is found
        if ball_x is not None:  # If ball position is found, stop checking further rows
            break

    # Print ball position
    if ball_x is not None:
        print(f"Ball detected at x position: {ball_x}")
    else:
        print("Ball not detected.")

    # Detect paddle position by searching only on the fixed row where the paddle resides
    for x in range(12, 148):  # Only check the middle 150 pixels, which is where the paddle usually spans
        if bwFrame[189, x] == 255:  # Look for white pixels (255) that represent the paddle
            paddle_x = x + 15  # Found a paddle pixel, record its x position plus an offset to center the paddle
            break  # Break out of the inner loop once the paddle is detected
        if paddle_x is not None:  # If a paddle has been found, stop searching further (this will crash the game if it's not detected as it is currently configured Dan: 11/5)
            break

    # Print paddle position
    if paddle_x is not None:
        print(f"Paddle detected at x position: {paddle_x}")
    else:
        print("Paddle not detected.")

    return ball_x, paddle_x  # Return the x-positions of the ball and paddle



# Checks if we should move the paddle to meet the ball depending on the coordinates of each
def getAction(ball_x, paddle_x):
    if (ball_x - paddle_x) > 5:
        return 2 # Move paddle right
    elif (paddle_x - ball_x) > 5:
        return 3 # Move paddle left
    else:
        return 0 # Do nothing

# Fires the first ball
obs, reward, terminated, truncated, info = env.step(1)



terminated = False
# Agent loop
while not terminated:
    # Find ball and paddle positions
    ball_x, paddle_x = findCoordinates(obs)
    if ball_x is None:
        ball_x = paddle_x #if ball is not found keep paddle still
    #Fire ball
    env.step(1)

    
    # Get the appropriate action
    action = getAction(ball_x, paddle_x)

    # Perform action
    #action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)


    if terminated or truncated:
        obs, info = env.reset()

env.close()
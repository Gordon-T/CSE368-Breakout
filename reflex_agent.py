import gymnasium as gym
import ale_py
import time
import numpy as np
import argparse

gym.register_envs(ale_py)

# Checks if we should move the paddle to meet the ball depending on the coordinates of each
def getAction(ball_x, paddle_x):
    if ball_x > paddle_x:
        return 2 # Move paddle right
    elif ball_x < paddle_x:
        return 3 # Move paddle left
    else:
        return 0 # Do nothing


def play(episodes, mode):
    render_mode = None
    if mode == True:
        render_mode='human'
    env = gym.make('ALE/Breakout-v5', render_mode=render_mode, obs_type="ram")#, frameskip=1)
    obs, info = env.reset()
    # Fires the first ball
    obs, reward, terminated, truncated, info = env.step(1)

    terminated = False
    # Agent loop

    scoreList = []
    for i in range(episodes):
        observation, info = env.reset()
        terminated = False
        while not terminated:
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
                print("Playing Progress: " + str(i) + "/" + str(episodes) + " games")
                scoreList.append(float(obs[77]))
                break
        
        env.close()
    # Playing metrics of the agent
    scoreSum = 0
    for i in scoreList:
        scoreSum += i
    scoreAvg = scoreSum / episodes
    print("---[Metrics]---")
    print("Average score: " + str(scoreAvg))
    print("Highest score out of the " + str(episodes) + " games: " + str(max(scoreList)))

def main():
    parser = argparse.ArgumentParser(description="A reflex agent for Atari Breakout")
    parser.add_argument('--games', type=int, default=10, help='Int; Specifies the amount of games you want to run the reflex agent on')
    parser.add_argument('--visible', default=False, action='store_true', help="True or False; Whether or not to show the game. Will get results slower if you do")
    args = parser.parse_args()

    # Access arguments
    if args.games and args.visible:
        print("---[Showing game window, this will make completing games slower since you have to actually see the game]---")
        print("---[Running " + str(args.games) + " games...]---")
        play(args.games, True)
    elif args.games and not args.visible:
        print("---[Not showing game window, launch with '--visible' parameter if you want to see the agent playing]---")
        print("---[Running " + str(args.games) + " games...]---")
        play(args.games, False)
    else:
        print("Error in arguments")

if __name__ == "__main__":
    main()
# CSE368 Final Project
A CSE 368 Project that uses machine learning and artifical intelligence concepts on Atari's _Breakout_. In this repository we have 2 agents:
- Reflex Agent: An agent that reads the game memory and reacts by moving the game paddle left or right depending on the position of the horizontal postion of the ball.
- Reinforcement Learning Agent: An agent that uses reinforcement training through the Q-learning algorithm and expands upon the former reflex agent. Through Q-learning, it will be rewarded for behavior that prevents losing through trying to match the paddle's horizontal position with the ball's and is penalized for deviating the paddle too far away from the ball's horizontal position.

This repository already includes a few trained agents in the ./q_tables folder that you can view and use for `rl_agent.py`.
- `breakout_qtable_#k.pkl` where # represents the amount of episodes x1000 that were used for training

# Pre-Requisite Libraries (Basically what we ran this on):
- Python 3.12.4
- [NumPy 2.2.0](https://numpy.org/install/)
- [Stella](https://stella-emu.github.io/)
- [Gymnasium 1.0.0](https://gymnasium.farama.org/)
- [ALE-Py 0.10.1](https://github.com/Farama-Foundation/Arcade-Learning-Environment)

# Usage
First, download this repository.
### Reflex Agent
To see the reflex agent play the game 10 times, run:
```
python reflex_agent.py --visible --games 10
```
- The `--visible` will make a visible game window appear. Removing this will speed up the process to get the metrics as you don't need to wait for each game to visually end.
- The `--games` argument is used to specify the amount of games you want the reflex agent to play as an integer. Default is 10 if this argument is not included.
- Once the agent is finished playing the specified amount of games, metrics such as the average score and highest score will be displayed in the terminal

### Reinforcement Learning Agent (Q-learning)
To train an agent over 3000 episodes of Breakout on your own machine, run:
```
python rl_agent.py --train --episodes 3000
```
- The `--train` argument is needed in order to train an agent, not including this argument will instead result in the program trying to play the game with an agent instead.
- The `--episodes` argument specifies the amount of episodes to train on as an integer. We would recommend a value of around 4000 which is also the default if this argument is not included
- By default `rl_agent.py` will save the q-table under the ./q_tables folder

To see the trained agent play a game, run:
```
python rl_agent.py --visible --games 10 --qfile breakout_qtable.pkl
```
- Do not include the `--train` argument if you want to see the agent play the game
- The `--visible` will make a visible game window appear. Removing this will speed up the process to get the metrics as you don't need to wait for each game to visually end.
- The `--games` argument is used to specify the amount of games you want the RL agent to play as an integer. Default is 10 if this argument is not included.
- The `--qfile` argument is used to specify a file that contains a saved q-table to load into the program under the ./q_tables folder. By default if not included, it will try to load `breakout_qtable.pkl`
- Once the agent is finished playing the specified amount of games, metrics such as the average score and highest score will be displayed in the terminal


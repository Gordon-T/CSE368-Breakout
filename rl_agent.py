import gymnasium as gym
import ale_py
import numpy as np
import random
import pickle
import os
import argparse

gym.register_envs(ale_py)

class QlearningAgent:
    def __init__(self, action_space, observation_space, learningRate, discountFactor, explorationRate, minExplorationRate, explorationDecay):

        self.action_space = action_space
        self.observation_space = observation_space
        
        # Hyperparameters, don't modify these here. If you want to modify these, you will have to go to main()
        self.learningRate = learningRate
        self.discountFactor = discountFactor
        self.explorationRate = explorationRate
        self.minExplorationRate = minExplorationRate
        self.explorationDecay = explorationDecay
        
        # q-table data structure
        self.q_table = {}
    
    # Save a q table using pickle for later use
    def save(self):
        os.makedirs('q_tables', exist_ok=True)
        filepath = os.path.join('q_tables', 'breakout_qtable.pkl') # TODO: If I got time, add an additional parameter for filename as well
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'explorationRate': self.explorationRate
            }, f)
        
        print(f"Q-table saved to {filepath}")
    
    # Load a q table using pickle
    def load(self, fileName):
        filepath = os.path.join('q_tables', fileName)
        
        try:
            with open(filepath, 'rb') as f:
                loaded_data = pickle.load(f)
                
            self.q_table = loaded_data['q_table']
            # TODO: Check if restore exploration rate is needed
            self.explorationRate = loaded_data.get(
                'explorationRate', 
                self.minExplorationRate
            )
            
            print(f"---[Q-table loaded from {filepath}]---")
            print(f"---[Loaded Q-table size: {len(self.q_table)} entries]---")
            print(f"---[Current exploration rate: {self.explorationRate:.4f}]---")
            return True
        except FileNotFoundError:
            print(f"---[No saved Q-table found at {filepath}]---")
            return False
        except Exception as e:
            print(f"---[Error loading Q-table: {e}]---")
            return False
        
    # Calculates a reward based off of the reflex agent.
    # observation: observation from the environment.
    # reward: Original reward from the environment.
    def calculateReward(self, observation, reward, prevObservation):
        updatedReward = reward  # Start with the original reward

        ball_x = observation[99]  # Ball X coordinate
        paddle_x = observation[72]  # Paddle X coordinate

        prev_ball_x = prevObservation[99] if prevObservation is not None else ball_x

        # Reward positioning paddle under the ball
        if abs(paddle_x - ball_x) < 2:  # fiddle with coordinate alignment?
            updatedReward += 0.5

        # Reward moving paddle closer to the ball's x coordinate
        if prevObservation is not None:
            if abs(prev_ball_x - paddle_x) > abs(ball_x - paddle_x): # fiddle with coordinate alignment?
                updatedReward += 0.05

        # TODO: Check if we can extract brick information from the observation state

        # Penalty if the paddle is farther away
        if prevObservation is not None:
            if abs(ball_x - paddle_x) > 15:
                updatedReward -= 0.1

        return updatedReward
    
    # Discretizes values
    def discretize(self, observation):
        return (
            self.discretizeValue(observation[99], 32), # Ball X coordinate
            self.discretizeValue(observation[72], 32), # Paddle X coordinate
            # TODO: Check if brick states are possible to obtain and if so, discretize them?
        )
    
    # Discretizes values
    def discretizeValue(self, value, bins):
        return min(int(value / (256 / bins)), bins - 1)
    
    # returns a q-value for a given state action pair
    # state: Current state
    # action: Action to evaluate
    def getQVal(self, state, action):
        return self.q_table.get((state, action), 0.0) #Look into second value more
    

    # The action selection with softmax activation function
    # state: Current discretized state
    def getAction(self, state):
        # Exploration
        if random.uniform(0, 1) < self.explorationRate:
            return self.action_space.sample()
        
        # Compute Q-values for all actions
        q_values = [self.getQVal(state, a) for a in range(self.action_space.n)]
        
        # Softmax action selection (with a bit of randomness)
        temperature = max(0.1, self.explorationRate)
        expValues = np.exp(np.array(q_values) / temperature)
        probs = expValues / np.sum(expValues)
        
        return np.random.choice(self.action_space.n, p=probs)
    
    # The core q-learning algorithm function
    # state: Current state
    # action: Chosen action
    # reward: Original reward received
    # nextState: Next state after taking the action
    # done: Whether the episode is finished
    # prevObservation: Previous raw observation (for reward shaping)
    # observation: Current raw observation (for reward shaping)
    def Q_learn(self, state, action, reward, nextState, done, prevObservation=None, observation=None):
        # Apply rewarding
        updatedReward = self.calculateReward(observation, reward, prevObservation)

        # Get current Q-value
        currentQ = self.getQVal(state, action)

        # Calculate maximum Q-value for next state
        if not done:
            nextQ_Values = [self.getQVal(nextState, a) for a in range(self.action_space.n)]
            maxNextQ = max(nextQ_Values)
        else:
            maxNextQ = 0

        # Q-learning update 
        newQ = currentQ + self.learningRate * (updatedReward + self.discountFactor * maxNextQ - currentQ)

        # Update Q-table
        self.q_table[(state, action)] = newQ

        # TODO: Gradual exploration decay, check if this affects average rewards
        self.explorationRate = max(self.minExplorationRate, self.explorationRate * self.explorationDecay)

# This function handles setting up the gymnasium environment and training an agent in that environment using q-learning
# episodes: How many games it should train for
# Other parameters are hyperparameters
def train(episodes, learningRate, discountFactor, explorationRate, minExplorationRate, explorationDecay):
    env = gym.make('ALE/Breakout-v5', render_mode=None, obs_type="ram")

    # Initialize agent object
    agent = QlearningAgent(
        action_space=env.action_space, 
        observation_space=env.observation_space,
        learningRate=learningRate,
        discountFactor=discountFactor,
        explorationRate=explorationRate,
        minExplorationRate=minExplorationRate,
        explorationDecay=explorationDecay
    )

    episodeRewards = []
    bestAvgReward = float('-inf')
    episodeAverages = []

    # Q-learning loop
    for episode in range(episodes):
        observation, info = env.reset()
        prevObservation = None

        # Discretize initial state
        state = agent.discretize(observation)

        totalReward = 0
        for step in range(1000): # Maximum allowed steps
            # Choose action and do it
            action = agent.getAction(state)
            nextObservation, reward, terminated, truncated, info = env.step(action)

            # Discretize next state
            nextState = agent.discretize(nextObservation)

            # Learn from the experience with reward
            agent.Q_learn(state, action, reward, nextState, terminated or truncated, prevObservation=prevObservation, observation=nextObservation)

            # Update state and accumulate reward
            state = nextState
            prevObservation = observation
            observation = nextObservation

            totalReward += reward

            if terminated or truncated:
                break

        # Print metrics for every 100 episodes
        episodeRewards.append(totalReward)
        if len(episodeRewards) >= 100:
            averageReward = np.mean(episodeRewards[-100:])
            if averageReward > bestAvgReward:
                bestAvgReward = averageReward
        if (episode + 1) % 100 == 0:
            averageReward = np.mean(episodeRewards[-100:])
            episodeAverages.append(float(averageReward))
            print("---[Training Progress]---")
            print(f"Episode {episode + 1}/{episodes}")
            print(f"Average Reward (last 100 episodes): {averageReward}")
            print(f"Exploration Rate: {agent.explorationRate:.4f}")
            print(f"Best 100-episode Average: {bestAvgReward}")
    print("Last 100 Reward Average History: " + str(episodeAverages))
    env.close()
    # Save the trained Q-table
    agent.save()
    return agent

# This function plays breakout with an (hopefully) trained agent
def play(agent, episodes, mode):
    render_mode = None
    if mode == True:
        print("---[Showing game window, this will make completing games slower since you have to actually see the game]---")
        render_mode='human'
    else:
        print("---[Not showing game window, launch with '--visible' parameter if you want to see the agent playing]---")
    scoreList = []
    env = gym.make('ALE/Breakout-v5', render_mode=render_mode, obs_type="ram")
    
    for i in range(episodes):
        observation, info = env.reset()
        state = agent.discretize(observation)
        totalReward = 0
        terminated = False
        
        while not terminated:
            # Choose best action based on learned with qlearning
            action = agent.getAction(state)
            observation, reward, terminated, truncated, info = env.step(action)
            # Update state
            state = agent.discretize(observation)
            totalReward += reward
            
            if terminated or truncated:
                print("Playing Progress: " + str(i) + "/" + str(episodes) + " games")
                scoreList.append(float(observation[77])) # Score offset
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
    print("List of scores: " + str(scoreList))

# Start of program, handle arguments and options from here
def main():
    parser = argparse.ArgumentParser(description="A Q-learning RL agent for Atari Breakout")
    parser.add_argument('--episodes', type=int, default=4000, help='Int; Specifies the amount of episodes you want to train the agent on')
    parser.add_argument('--games', type=int, default=10, help='Int; Specifies the amount of games you want to run the agent on')
    parser.add_argument('--train', default=False, action='store_true', help="Whether to train or play the game")
    parser.add_argument('--visible', default=False, action='store_true', help="Whether or not to show the game. Will get results slower if you do")
    parser.add_argument('--qfile', type=str, default='breakout_qtable.pkl', required=False, help="Specify a .pkl q-table file under the ./q_tables folder to play or train on")
    args = parser.parse_args()

    if args.train == True: # Train a model
        print("---[Starting Q-learning training for " + str(args.episodes) + " episodes]---")
        agent = train(args.episodes, 0.01, 0.975, 1.0, 0.05, 0.99)
        print("---[Training complete, run this file again without the '--train' parameter and with the '--visible' parameter to see the agent play]---")
    elif args.train == False: # The default option, will try to load a q-table and play
        print("---[Loading q-table from disk and playing on " + str(args.games) + " games]---")
        print("---[Searching for " + str(args.qfile) + ", use '--qfile' parameter to specify file name]")
        env = gym.make('ALE/Breakout-v5', obs_type="ram")
        agent =  QlearningAgent(env.action_space, env.observation_space, 0.01, 0.975, 1.0, 0.05, 0.99)
        agent.load(args.qfile)
        play(agent, args.games, args.visible)
    else:
        print("Error in arguments")

if __name__ == "__main__":
    main()
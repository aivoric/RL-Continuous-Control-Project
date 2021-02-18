"""
### main.py ###
- Entry point into training the algorithm.
- Import and create instances of the environment (Reacher) and Agent, then 
pass then on into a training function.
"""

import numpy as np
import torch
from collections import deque
import time
from os import path
import pickle
from unityagents import UnityEnvironment
from model import ActorCriticNetwork
from agent import Agent

# Load Reacher environment
env = UnityEnvironment(file_name="Reacher.app", seed=1, no_graphics=True)

# Get the brain and reset environment
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]

# Check the number of agents:
num_agents = 20
print('Number of agents:', num_agents)

# Check the size of actions:
action_size = brain.vector_action_space_size
print('Action size::', action_size)

# Check the size of each state:
state_size = brain.vector_observation_space_size
print('State size::', state_size)
    
def a2c(agent, num_agents, num_episodes=300):
    """
    Training loop which runs across episodes and collects scores.
    """
    all_scores = []
    scores_window = deque(maxlen=100)
    first_solve = True
    best_score = 30

    for episode in range(1, num_episodes + 1):

        avg_score = agent.step()
        scores_window.append(avg_score)
        score = np.mean(scores_window)
        all_scores.append(avg_score)
        
        if episode % 5 == 0:
            print("Episode {} score: {}".format(episode, score))

        if score >= best_score:
            best_score = score
            if first_solve:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, score))
                first_solve = False
            else:
                print('\nAchieved a better score of {:.2f} after {:d} episodes'.format(score, episode))
            torch.save(agent.network.state_dict(), path.join('models', 'solution'))

    return all_scores

# Create the agent and then run the agent through the a2c algorithm:
agent = Agent(env, brain_name, num_agents, state_size, action_size)
scores = a2c(agent, num_agents)

# Save the results:
# TODO: Refactor this later to include various versions for experimentation:
experiment_name = "main"
timestamp = int(time.time())
results_file_name = path.join('results', '{}_{}_results'.format(timestamp, experiment_name))
model_file_name = path.join('models', '{}_{}'.format(timestamp, experiment_name))

with open(results_file_name, 'wb') as handle:
    pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("\nResults saved to: {}.".format(results_file_name))
    
torch.save(agent.network.state_dict(), model_file_name)
print("\nModel saved to: {}.".format(model_file_name))

# Close the unity environment
env.close()

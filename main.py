import numpy as np
from unityagents import UnityEnvironment
import torch
from collections import deque
from random import randint

# This is a non-blocking call that only loads the environment.
env = UnityEnvironment(file_name="Reacher.app", seed=1, no_graphics=False)

scores = []
scores_window = deque(maxlen=100)
best_score = -100
model_state_dict = {}

# Get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# Get number of actions and state size
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
state_size = len(state)
max_steps = 1000

for step in range(max_steps):
    
    # Move the environment to the next state with the appropriate action
    env_info = env.step(action)[brain_name]             
    
    # Get the next state, reward, and whether the state is done
    next_state = env_info.vector_observations[0]        
    reward = env_info.rewards[0]                        
    done = env_info.local_done[0]
    
    # Update state
    state = next_state
    
    # Update the score:
    episode_score += reward
    if done:
        break

# https://github.com/JoshVarty/Reacher/blob/master/main.py

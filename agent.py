#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

"""
Contains 2 classes: Agent and Batcher.
The Agent interacts with the environment, collects data, and trains.
The Batcher assists the Agent with handling the data in batches for training.
"""

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from model import ActorCriticNetwork

class Agent:
    """
    Agent is responsible for walking through the environment episodes (step() method) while
    collecting states, actions taken, and rewards received.
    The Agent also interacts with the Batcher class which helps handle the collected
    data for training.
    The Agent finally trains the network by going through all the batches of data, calculating the loss
    and then backpropagates. This is handled by the train_network() method.
    """
    def __init__(self, environment, brain_name, num_agents, state_size, action_size):
        self.environment = environment
        self.brain_name = brain_name
        self.num_agents = num_agents
        self.network = ActorCriticNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.network.parameters(), 2e-4, eps=1e-5)

        # Define all the hyperparameters
        self.discount_rate = 0.99
        self.tau = 0.95
        self.learning_rounds = 10
        self.ppo_clip = 0.2
        self.gradient_clip = 5
        self.mini_batch_number = 64


    def generate_rollout(self):
        """
        The first method called during a step() through the environment.
        Collects episode data, and returns it for further processing.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        rollout = []
        episode_rewards = np.zeros(self.num_agents)

        #Reset environment
        env_info = self.environment.reset(train_mode=True)[self.brain_name]
        states = env_info.vector_observations
        
        #Generate rollout from an entire episode
        while True:

            states = torch.Tensor(states).to(device)
            actions, log_probs, values = self.network(states)
            env_info = self.environment.step(actions.cpu().detach().numpy())[self.brain_name]
            rewards = env_info.rewards
            dones = np.array(env_info.local_done)
            next_states = env_info.vector_observations

            episode_rewards += rewards
            rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), rewards, 1 - dones])
            states = next_states

            if np.any(dones):
                break

        states = torch.Tensor(states).to(device)
        _,_,last_value = self.network(states)
        rollout.append([states, last_value, None, None, None, None])
        return rollout, last_value, episode_rewards

    def process_rollout(self, rollout, last_value):
        """
        The second method called during a step() through the environment.
        Receives the data from an episode and processes it to prepare for training,
        e.g. flattens the data, calculates advantages, error, etc.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        processed_rollout = [None] * (len(rollout) - 1)
        advantages = torch.zeros((self.num_agents, 1)).to(device)
        returns = last_value.detach()

        for i in reversed(range(len(rollout) - 1)):
            states, value, actions, log_probs, rewards, dones = rollout[i]
            dones = torch.Tensor(dones).unsqueeze(1).to(device)
            rewards = torch.Tensor(rewards).unsqueeze(1).to(device)
            next_value = rollout[i + 1][1]
            returns = rewards + self.discount_rate * dones * returns

            td_error = rewards + self.discount_rate * dones * next_value.detach() - value.detach()
            advantages = advantages * self.tau * self.discount_rate * dones + td_error
            processed_rollout[i] = [states, actions, log_probs, returns, advantages]

        return processed_rollout

    def train_network(self, states, actions, log_probs_old, returns, advantages):
        """
        The third method called during a step() through the environment.
        Instantiates a Batcher class for handling batches of shuffled data.
        Iterates through the batches until empty.
        For each batch calculates the loss, and then backpropagates to train the network.
        """
        batcher = Batcher(states.size(0) // self.mini_batch_number, [np.arange(states.size(0))])
        for _ in range(self.learning_rounds):
            batcher.shuffle()
            while not batcher.end():
                batch_indices = batcher.next_batch()[0]
                batch_indices = torch.Tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                _, log_probs, values = self.network(sampled_states, sampled_actions)
                ratio = (log_probs - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean(0)

                value_loss = 0.5 * (sampled_returns - values).pow(2).mean()

                self.optimizer.zero_grad()
                (policy_loss + value_loss).backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip)
                self.optimizer.step()

    def step(self):
        """
        Handles the agent stepping through the environment.
        Calls various class methods which:
        1) Collect data, 2) Process data, 3) Train network
        Returns the average score.
        """
        rollout, last_value, episode_rewards = self.generate_rollout()
        processed_rollout = self.process_rollout(rollout, last_value)
        states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
        advantages = (advantages - advantages.mean()) / advantages.std()
        self.train_network(states, actions, log_probs_old, returns, advantages)
        return np.mean(episode_rewards)


class Batcher:
        """
        Helps organise all the collected data into batches, 
        and contains various helper methods for e.g. shuffling the data.
        """
    def __init__(self, batch_size, data):
        self.batch_size = batch_size
        self.data = data
        self.num_entries = len(data[0])
        self.reset()

    def reset(self):
        """
        Resets the entire batch.
        """
        self.batch_start = 0
        self.batch_end = self.batch_start + self.batch_size

    def end(self):
        return self.batch_start >= self.num_entries

    def next_batch(self):
        """
        Returns the next batch of data which is dependent on the batch_size.
        """
        batch = []
        for d in self.data:
            batch.append(d[self.batch_start: self.batch_end])
        self.batch_start = self.batch_end
        self.batch_end = min(self.batch_start + self.batch_size, self.num_entries)
        return batch

    def shuffle(self):
        """
        Shuffles the data in order to remove unnecessary correlations
        and improve training.
        """
        indices = np.arange(self.num_entries)
        np.random.shuffle(indices)
        self.data = [d[indices] for d in self.data]
        self.reset()
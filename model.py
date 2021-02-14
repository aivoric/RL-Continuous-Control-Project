import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNetwork(nn.Module):

    def __init__(self, state_size, action_size, output=None):

        super(FCNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)
        self.output = output

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        if self.output is not None:
            x = self.output(x)

class ActorCriticNetwork(nn.Module):

    def __init__(self, state_size, action_size):
        super(ActorCriticNetwork, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.actor = FCNetwork(state_size, action_size, torch.tanh)
        self.critic = FCNetwork(state_size, 1)
        self.std = nn.Parameter(torch.ones(1, action_size)).to(self.device)
        self.to(self.device)

    def forward(self, state, action = None):
        
        a = self.actor(state)
        distribution = torch.distributions.Normal(a, self.std)
        if action is None:
            action = distribution.sample()
        log_prob = distribution.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)

        #Get value from critic
        value = self.critic(state)

        return action, log_prob, value
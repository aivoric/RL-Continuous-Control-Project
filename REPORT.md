## Summary

### Results

The environment was solved after approximately 75 episodes.

A graphic summarising the performance:
!["Model Results"](https://github.com/aivoric/RL-Continuous-Control-Project/blob/main/results.png?raw=true)

The solution weight can be found in the /models folder. File called: solution.

For even better weights (more training) check out the 1612720741_main file.

### Learning Algorithm

The algorithm used to solve the environment is PPO using an older implementation from Shangtong Zhang.

It is broken down into the following steps which are executed by the agent's step() function:
- Run an episode and collect data for 20 agents, where each episode is 1000 steps. The data is appended into a rollout list and returned.
- The data (rollouts) are then processed where the discounted returns and the advantages are calculated. Processed data is stored in tuples and returned.
- The networks are then trained using these returned tuples. The final score is then returned.


### Model Architecture

The architecture is a simple Linear model for both the Actor and the Critic:
- Consumes the state size
- 1 inner layer with 256 input and output neurons
- Outputs the action size
- Relu activation function is used in each layer (apart from the output)
- The actor has one difference though: there is an additional Tanh activation function for the output

### Hyperparameters

- discount rate = 0.99
- tau = 0.95
- learning rounds = 10
- ppo clip = 0.2  (the same one specified in the oririginal paper)
- gradient clip = 5
- batch size = 64

## Future Improvements

Further improvements should include:

__Network architectures__ 
Currently a very simple network architecture is used. In many cases this is enough, however at 
times trying a network architecture with more layers and more neurons can yield better results.

__Try other algorithms__
I would try TRPO and DDPG as alternative learning algorithms. It's difficult to know which learning 
algorithm would work best for this task. PPO seemed like a good choice for this task, but intuition says
that DDPG could achieve better results.

__Try other hyperparameters__
The current set of hyperparameters were determined mostly based on intuition from previous work. 
I would like to especially experiment with setting different tau, batch sizes, and learning rounds.

__Try more agents__
In the exercise we trained with 20 agents, but what if more simultaneous agents were used? Would that 
improve training? This would be a good hypothesis to test.

__Learn from real life__
The final improvement is to try move this virtual simulation to a real robotic arm, and use the algorithm
there, or at least a similar version of it depending on whether the state and action sizes vary.
## Introduction

This is a deep reinforcement learning project which solves the Reacher unity environment.

An example of the reacher environment:
!["Reacher Example"](https://github.com/aivoric/RL-Continuous-Control-Project/blob/main/reacher-example.gif?raw=true)

Enjoy the read below for more details!

## Environment Introduction

The environment is a 3D space where a double-jointed arm can move to a target location.

The goal is for the arm (agent) to move to the goal location and keep it there.

The environment contains 20 agents with the same behaviour parameters.

Reward function:
The agent receive a reward of +0.1 for every step the agent's hand is in goal location.

Behavior Parameters:
- Vector Observation space: 26 variables corresponding to position, rotation, velocity, and angular velocities of the two arm rigid bodies.
- Actions: 4 continuous actions, corresponding to torque applicable to two joints.
- Visual Observations: None.

## How to Install the Project

To setup the environment on your machine:
1. Install Python 3.8
2. Clone this repository:
        git clone https://github.com/aivoric/RL-Continuous-Control-Project.git
3. Create a virtual python environment:
        python3 -m venv venv
4. Activate it with:
        source venv/bin/activate
5. Install all the dependecies:
        pip install -r requirements.txt
6. Download and install Anaconda so that you can run a Jupyter notebook from:
        http://anaconda.com/

## Overview of Project Files

The project is broken down into the following core python files:
- main.py
- agent.py
- model.py

The following 2 folders:
- /models  (the solution model weights are found in here)
- /results (the score for each episode is stored here)

And the following jupyter notebook:
- Continuous Control Results.ipynb

All the other files are used by the unity environment which allow you to run the environment. Most of the files are based on version 0.4 of ml-agents which is from July 2018 so it is quite outdated. For reference, a more modern ml-agents can be downloaded from: 
https://github.com/Unity-Technologies/ml-agents 

## Training

To train the model run the main.py file.
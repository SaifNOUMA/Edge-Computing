#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import argparse
import tensorflow as tf
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow import keras
from ns3gym import ns3env

__author__ = "Piotr Gawlowicz"
__copyright__ = "Copyright (c) 2018, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "gawlowicz@tkn.tu-berlin.de"

parser = argparse.ArgumentParser(description='Start simulation script on/off')
parser.add_argument('--start',
                    type=int,
                    default=1,
                    help='Start ns-3 simulation script 0/1, Default: 1')
parser.add_argument('--iterations',
                    type=int,
                    default=1,
                    help='Number of iterations, Default: 1')
args = parser.parse_args()
startSim = bool(args.start)
iterationNum = int(args.iterations)

# Connect Gym With NS3 Simulation:
port = 5555
simTime = 20  # seconds
stepTime = 0.5  # seconds
seed = 0
simArgs = {"--simTime": simTime,
           "--testArg": 123}
debug = False
env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)
env.reset()


ob_space = env.observation_space
ac_space = env.action_space
print("Observation space: ", ob_space, ob_space.dtype)
print("Action space: ", ac_space, ac_space.dtype)


# Define The HyperParameters For Training The Agent :
s_size = ob_space.shape[0]
a_size = ac_space.n

max_env_steps = 300
env._max_episode_steps = max_env_steps
epsilon = 1.0  # exploration rate
epsilon_min = 0.01
epsilon_decay = 0.999


# Implement an Agent Which is a Fully Connected Neural Network
model = keras.Sequential()
model.add(keras.layers.Dense(s_size, input_shape=(s_size,), activation='relu'))
model.add(keras.layers.Dense(12, activation = "relu"))
model.add(keras.layers.Dense(a_size, activation='softmax'))
model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])




time_history = []
rew_history = []
observations = []

for episode in range(30):
    print("Episode : %d" % episode)

    rewardsum = 0
    state = env.reset()
    state = np.reshape(state, [1, s_size])

    for time in range(max_env_steps):
        print("Step : %d  |||  Episode : %d" % (time, episode))
        index = np.argmax(state)
        
        # Choose action using epsilon-Greedy Policy
        if np.random.rand(1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(state)[0])


        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, s_size])
        

        # Train The Agent :
        target = reward
        if not done:
            target = (reward + 0.90 * np.amax(model.predict(next_state)[0]))
        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)


        state = next_state
        rewardsum += reward
        if epsilon > epsilon_min: epsilon *= epsilon_decay
    
    time_history.append(time)
    rew_history.append(rewardsum)



# Save The Reward Values During All The Iterations in a  pickle file:
dic = {"time" : time_history,
        "reward" : rew_history }
path_file = "scratch/offloading-v0/result.pickle"
with open(path_file, 'wb') as f:
    pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)



# Plot The Learning Performance Figure :
print("Plot Learning Performance")
mpl.rcdefaults()
mpl.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(figsize=(10, 4))
plt.grid(True, linestyle='--')
plt.title('Learning Performance')
plt.plot(range(len(rew_history)), rew_history, label='Reward', marker="", linestyle="-")  # , color='k')
plt.xlabel('Episode')
plt.ylabel('Time')
plt.legend(prop={'size': 12})

plt.savefig('learning.pdf', bbox_inches='tight')
plt.show()

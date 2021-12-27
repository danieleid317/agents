'''
This is an unbatched implementation of
reinforce with no reward normalization, advantage , and no entropy bonus.
Tensorflow is used to build the actor network , and a custom built environment
is used to simulate a cartpole in PyBullet.

The agent learns to solve the environment within a few hundred episodes.

Completed Objectives:
Building of discrete RNN actor network.
Building of custom environment using PyBullet.
Application of REINFORCE algorithm in custom training loop.
'''

import numpy as np
import random
import statistics
import tensorflow as tf

from matplotlib import pyplot as plt
from tensorflow.keras import layers

import numpy as np
import pybullet as p
import pybullet_data

import time

from tensorflow import keras
from envs import DiscretePyBulletCartPoleEnv

env = DiscretePyBulletCartPoleEnv()

model = keras.Sequential()
model.add(layers.Input(shape=(None,4,)))
# The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
model.add(layers.GRU(64))

model.add(layers.Dense(2))
model.compile()


def calc_g( reward_trajectory , gamma ):
  ez_discount = np.array([ gamma**n for n in range(len(reward_trajectory))])
  gs = []
  reward_trajectory = np.array(reward_trajectory)
  for ts in range(len(reward_trajectory)):
    to_end_rewards = reward_trajectory[ts:]
    eq_len_discount = ez_discount[:len(reward_trajectory[ts:])]
    total_value = np.multiply(to_end_rewards , eq_len_discount )
    g = sum(total_value)
    gs.append(g)
  return gs

def step_episode(env , model ):
  env.reset()
  states = []
  action_probs_list = []
  rewards = []
  playing = True
  while playing == True:
    obs = env.state.tolist()
    states.append(obs)
    #print('obs ' , obs)
    #print(tf.expand_dims( states , 0))
    #run model to get action logits and value
    action_logits  = model( tf.expand_dims( states , 0) )
    #categorical probabilistic action idx selection
    selected_action_idx = tf.random.categorical( action_logits , 1 )[ 0 , 0 ]
    reward , playing = env.step(selected_action_idx)
    #normalized probs
    action_probs = tf.nn.softmax(action_logits)
    probability_of_taking_selected_action = action_probs[0 , selected_action_idx]
    action_probs_list.append(probability_of_taking_selected_action)
    rewards.append(reward)
  return action_probs_list , rewards

def actor_loss( action_probs , rewards ):
  # log of pi (a | s)
  gs = calc_g(rewards, .999)
  action_log_probs = tf.math.log(action_probs)
  actor_loss =  - tf.math.reduce_sum(action_log_probs * gs)
  return actor_loss

optimizer= tf.keras.optimizers.Adam(learning_rate=.0005)

average_reward = []
for episode in range(1000):
  with tf.GradientTape() as tape:
    action_probs , rewards = step_episode( env , model)
    loss = actor_loss( action_probs , rewards )
  grads = tape.gradient(loss , model.trainable_variables )
  optimizer.apply_gradients(zip(grads , model.trainable_variables))
  average_reward.append(sum(rewards))#adds episode average reward
  if episode % 5 == 0:
      print(np.mean(average_reward[-10:]))

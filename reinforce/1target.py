'''
this is an unbatched implementation of
reinforce with no reward normalization, advantage , and no entropy bonus

The environment is a simple 7x7 2d gridworld and the objective is for the agent to
reach the goal. The agent and target position are randomly initialized every episode.
The metric being monitored to indicate learning is the average number of steps it
takes the agent to reach the goal.
'''

import numpy as np
import random
import statistics
import tensorflow as tf

from matplotlib import pyplot as plt
from tensorflow.keras import layers

import numpy as np
import time

from envs import OneTargetGridworld

env = OneTargetGridworld()

class Actor(tf.keras.Model):
  def __init__( self , num_actions , num_hidden_units ):
    super().__init__()

    self.shared_1 = layers.Dense( num_hidden_units , activation ='relu')
    self.actor = layers.Dense(num_actions)

  def call( self , input_obs ):
    x = self.shared_1(input_obs)
    return self.actor(x)

model = Actor( num_actions = 4 , num_hidden_units = 100 )

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
  action_probs_list = []
  rewards = []
  playing = True
  while playing == True:
    obs = tf.expand_dims( env.state , 0 )
    #run model to get action logits and value
    action_logits  = model(obs)
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

average_len = []
for episode in range(5000):
  with tf.GradientTape() as tape:
    action_probs , rewards = step_episode( env , model)
    loss = actor_loss( action_probs , rewards )
    grads = tape.gradient(loss , model.trainable_variables )
    optimizer.apply_gradients(zip(grads , model.trainable_variables))
    average_len.append(len(rewards))#adds episode average reward
    if episode % 50 == 0:
      print('Episode : ' , episode )
      print('Average steps to target ' , np.mean(average_len[-100:]))

''' The length of the trajectories decreases as training continues indicating
that the agent is taking fewer steps per episode to reach the target'''

'''
This is a complete implementation of REINFORCE using mean rewards to calculate
an advantage, and also ading an entropy bonus for better exploration.

The gridworld environment expects the agent to go to both targets in their respective order
This is done through the use of memory thanks to an RNN network, because the environment does not provide
information about which targets have been reached.
'''

import numpy as np
import random
import statistics
import tensorflow as tf

from matplotlib import pyplot as plt
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
import time
from collections import deque

from envs import TwoTargetGridworld

env = TwoTargetGridworld()

model = keras.Sequential()
model.add(layers.Input(shape=(None,6,)))
# The output of GRU will be a 3D tensor of shape (batch_size, 256)
model.add(layers.GRU(256))
num_actions = 4
model.add(layers.Dense( 32 ))
model.add(layers.Dense(num_actions))
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

mean_buffer_rewards = deque( [] , maxlen = 10000)

class MeanAdvantageBuffer:
    def __init__(self):
        self.rewards = []

    def add_reward( self , reward ):
        self.rewards.append( reward )

    def calc_advantage(self , reward):
        pass

def step_episode(env , model ):
  env.reset()
  action_probs_list = []
  normalized_rewards = []
  playing = True
  states = []
  entropy = 0
  while playing == True:
    states.append(env.state)
    #run model to get action logits and value
    action_logits  = model(tf.expand_dims( states , 0 ))
    #categorical probabilistic action idx selection
    selected_action_idx = tf.random.categorical( action_logits , 1 )[ 0 , 0 ]
    reward , playing = env.step(selected_action_idx)

    #normalized probs
    action_probs = tf.nn.softmax(action_logits)
    log_action_probs = tf.math.log(action_probs)
    entropy += -1 * tf.reduce_sum(action_probs * log_action_probs)
    probability_of_taking_selected_action = action_probs[0 , selected_action_idx]
    action_probs_list.append(probability_of_taking_selected_action)

    mean_buffer_rewards.append( reward )
    mean_reward = statistics.mean(mean_buffer_rewards)
    normalized_reward = reward - mean_reward
    normalized_rewards.append( normalized_reward )

  gs = calc_g(normalized_rewards, .98 )
  return action_probs_list , gs , sum( normalized_rewards ) , entropy


def make_batch( num_episodes ):
    batch_actions = []
    batch_gs = []
    rewards = 0
    entropy = 0
    for episode in range(num_episodes):
        action_probs , gs , reward , entropy = step_episode( env , model )
        batch_actions = batch_actions + action_probs
        batch_gs = batch_gs + gs
        rewards += reward
        entropy += entropy
    ave_reward = rewards / num_episodes
    return batch_actions , batch_gs , ave_reward , entropy

entropy_beta = 0.12

def actor_loss( action_probs , gs , entropy ):
  # log of pi (a | s)
  action_log_probs = tf.math.log(action_probs)
  entropy_reg = entropy * entropy_beta
  actor_loss =  - tf.math.reduce_sum(action_log_probs * gs) - entropy_reg
  return actor_loss

optimizer= tf.keras.optimizers.Adam(learning_rate=.0005)

ave_len = deque([] , maxlen=100 )
batch_size = 4
rew = deque([] , maxlen=100 )
for episode in range(5000):
  with tf.GradientTape() as tape:
    action_probs , gs , adv_reward , entropy = make_batch( batch_size )
    loss = actor_loss( action_probs , gs , entropy )
  grads = tape.gradient(loss , model.trainable_variables )
  optimizer.apply_gradients(zip(grads , model.trainable_variables))
  ave_len.append(len(gs)/batch_size)#adds episode average reward
  rew.append( adv_reward )
  if episode % 25 == 0:
    print(episode , ' length   ' ,  statistics.mean(ave_len))
    print(episode , ' reward   ' ,  statistics.mean(rew))

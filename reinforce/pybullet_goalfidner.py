'''
This is a vanilla reinforce agent that uses average reward as a baseline.
Thanks to mini batches this algo can usually take steps in the correct direction arriving at a local
optima.  The environment is a goal finding for pybullet using discrete predictions, entropies are also
added for exploration because he agent is navigating through a 3-D gridworld vs 2-D.
'''

import time
import tensorflow as tf
import numpy as np
import random
import pybullet as p
import pybullet_data
import statistics


from envs import PyBulletGoalFinder

entropy_beta = .05

baseline_rewards = []

env = PyBulletGoalFinder()

class Actor(tf.keras.Model):
  def __init__( self , num_actions , num_hidden_units ):
    super().__init__()

    self.shared_1 = tf.keras.layers.Dense( num_hidden_units , activation ='relu')
    self.actor = tf.keras.layers.Dense(num_actions)

  def call( self , input_obs ):
    x = self.shared_1(input_obs)
    return self.actor(x)

model = Actor( num_actions = 6 , num_hidden_units = 16 )

def calc_g( reward_trajectory , gamma ):
  ez_discount = np.array([ gamma**n for n in range(len(reward_trajectory))])
  gs_list = []
  reward_trajectory = np.array(reward_trajectory)
  for ts in range(len(reward_trajectory)):
    to_end_rewards = reward_trajectory[ts:]
    eq_len_discount = ez_discount[:len(reward_trajectory[ts:])]
    total_value = np.multiply(to_end_rewards , eq_len_discount )
    g = sum(total_value)
    gs_list.append( g )
  return gs_list

def episode_batch(env , model , num_episodes ):
    action_probs_list = []
    gs_list = []
    entropies_list = []
    for episode in range(num_episodes):
        action_probs , gs , entropy_losses= step_episode( env , model )
        action_probs_list.append(action_probs)
        gs_list.append( gs )
        entropies_list.append( entropy_losses )
    return action_probs_list  , gs_list , entropies_list

def step_episode(env , model ):
  env.reset()
  action_probs_list = []
  rewards = []
  advbase = []
  entropy_losses = []
  playing = True
  while playing == True:
    obs = tf.expand_dims( env.state.tolist() , 0 )
    #run model to get action logits and value
    action_logits  = model(obs)
    #categorical probabilistic action idx selection
    selected_action_idx = tf.random.categorical( action_logits , 1 )[ 0 , 0 ]
    reward , playing = env.step(selected_action_idx.numpy())
    #normalized probs
    action_probs = tf.nn.softmax(action_logits)
    probability_of_taking_selected_action = action_probs[0 , selected_action_idx]
    action_probs_list.append(probability_of_taking_selected_action)
    entropy_loss = - entropy_beta * (sum(action_probs[0] * np.log(action_probs[0])))
    entropy_losses.append(entropy_loss)
    rewards.append(reward)
    baseline_rewards.append(reward)
  bla = statistics.mean(rewards[-20000:])
  for rew in rewards:
      advbase.append(rew - bla)

  gs = calc_g(advbase, .999)
  #gs = ((gs - tf.math.reduce_mean(gs)) / (tf.math.reduce_std(gs) ))
  return action_probs_list , gs , entropy_losses

def actor_loss( action_probs , gs , entropies ):
  # log of pi (a | s)
  action_log_probs = tf.math.log(action_probs)
  actor_loss =  - tf.math.reduce_sum(action_log_probs * gs)
  print('actor loss ' , actor_loss , 'entropy' , sum(entropies))
  return actor_loss +  sum(entropies)

optimizer= tf.keras.optimizers.Adam(learning_rate=.005)

#training setup below

def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

average_reward = []
average_steps_per_episode = []
episodes_per_batch = 16
for batch in range(2000):
  with tf.GradientTape() as tape:
    action_probs , gs , entropies = episode_batch( env , model , episodes_per_batch )
    gs = flatten_list(gs)
    action_probs = flatten_list(action_probs)
    entropies = flatten_list(entropies)
    average_steps_per_episode.append(len(gs)/episodes_per_batch)
    loss = actor_loss( action_probs , gs , entropies )
    grads = tape.gradient(loss , model.trainable_variables )
    optimizer.apply_gradients(zip(grads , model.trainable_variables))
    average_reward.append(sum(gs))#adds episode average reward
    if batch % 5 == 0:
      print(batch , np.mean(average_steps_per_episode[-20:]) , 'steps per episode average' ,)

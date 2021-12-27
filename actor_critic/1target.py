'''
Advantage actor critic no entropy
'''

import numpy as np
import random
import statistics
import tensorflow as tf

from matplotlib import pyplot as plt
from tensorflow.keras import layers

import time

class TargetGridworld():
  def __init__(self):
    self.state = self.init_state()
    self.step_count = 0
    self.first = 0
  def init_state(self):
    xs = random.sample(range(0,10),5)
    ys = random.sample(range(0,10),5)
    piece1 = [xs[0],ys[0]]

    playerpos = [xs[4],ys[4]]
    positions = list(np.array([ playerpos , piece1 ]).flatten())

    return positions

  def reset(self):
    self.state = self.init_state()
    self.step_count = 0

  def step(self, action):
    self.step_count += 1
    if (self.step_count >= 150 ) : # loss maxsteps
      self.reset()
      reward =   - 1
      playing = False
      return reward , playing

    if ( self.state[0] == self.state[2] ) and ( self.state[1] == self.state[3] ): # first checkpoint
      playing = False
      reward = 100
      self.reset()
      return reward , playing

    if action == 0:#up
        self.state[1] -= 1

    if action == 1:#right
        self.state[0] += 1

    if action == 2:#down
        self.state[1] += 1

    if action == 3 :#left
        self.state[0] -= 1

    playing = True
    reward =  -1
    return reward , playing

env = TargetGridworld()

class ActorCritic(tf.keras.Model):
  def __init__( self , num_actions , num_hidden_units ):
    super().__init__()

    self.shared_1 = layers.Dense( num_hidden_units , activation ='relu')
    self.actor = layers.Dense(num_actions)
    self.critic = layers.Dense(1)

  def call( self , input_obs ):
    x = self.shared_1(input_obs)
    return self.actor(x) , self.critic(x)

model = ActorCritic( num_actions = 4 , num_hidden_units = 100 )

def step_episode(env , model ):
  env.reset()
  action_probs_list = []
  pred_values = []
  target_vals = []
  advantages = []
  playing = True
  while playing == True:
    obs = tf.expand_dims( env.state , 0 )
    #run model to get action logits and value
    action_logits , pred_value  = model(obs)
    pred_values.append( pred_value[0][0] )
    #categorical probabilistic action idx selection
    selected_action_idx = tf.random.categorical( action_logits , 1 )[ 0 , 0 ]
    reward , playing = env.step(selected_action_idx)
    _ , next_value  = model(tf.expand_dims( env.state , 0 ), training=False)

    target_value = reward + ( next_value[0][0] * .95 )
    advantages.append( target_value - pred_value )
    target_vals.append( target_value )
    #normalized probs
    action_probs = tf.nn.softmax(action_logits)
    probability_of_taking_selected_action = action_probs[0 , selected_action_idx]
    action_probs_list.append(probability_of_taking_selected_action)
  return action_probs_list , target_vals , pred_values , advantages

def actor_loss( action_probs , advantages ):
  # log of pi (a | s)
  action_log_probs = tf.math.log(action_probs)
  print(action_log_probs)
  #print(advantages)
  actor_loss =  - tf.math.reduce_sum(action_log_probs * advantages ) / len(advantages)
  print(actor_loss)
  return actor_loss

huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
def critic_loss( targets , predictions ):
  l = huber_loss( targets , predictions ) / len(targets)
  return l

def actor_critic_loss(action_probs , targets , predictions , adv ):
  actor_loss_val = actor_loss(action_probs , adv )
  critic_loss_val = critic_loss( targets , predictions )
  return actor_loss_val + critic_loss_val

optimizer= tf.keras.optimizers.Adam(learning_rate=.0002)

average_len = []
for episode in range(50000):
  with tf.GradientTape() as tape:
    action_probs , targets , predictions , adv = step_episode( env , model)
    loss = actor_critic_loss( action_probs , targets , predictions , adv )
    grads = tape.gradient(loss , model.trainable_variables )
    optimizer.apply_gradients(zip(grads , model.trainable_variables))
    average_len.append(len(adv))#adds episode average reward
    if episode % 50 == 0:
      print(episode , np.mean(average_len[-100:]))

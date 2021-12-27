'''
Actor critic with no baseline advantage and no entropy
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

class PyBulletCartPoleEnv():
  def __init__(self):
    self.state = self.init_state()
    self.step_count = 0

  def init_state(self):
    p.connect(p.DIRECT)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-9.8)
    p.loadURDF("cartpole.urdf" , [0 , 0 , 1 ] , [ 0 , 0 , 0 , 1 ])
    p.setJointMotorControl2(0, 0, p.VELOCITY_CONTROL, force=0)
    p.setJointMotorControl2(0, 1, p.VELOCITY_CONTROL, force=0)
    obs = np.array(p.getJointStates( 0 , [0 , 1] ))[: , :2].flatten()
    return obs

  def reset(self):
    p.disconnect()
    self.state = self.init_state()
    self.step_count = 0

  def step(self, action):
    self.step_count += 1

    if (self.step_count >= 200) :
      self.reset()
      reward =   1
      playing = False
      return reward , playing

    if (abs(self.state[0]) > 3) :#moves too far outside horizontaly on cart
      self.reset()
      playing = False
      reward = 0
      return reward , playing

    if abs(self.state[2]) >= .2 :# not within angle
      self.reset()
      playing = False
      reward = 0
      return reward , playing

    if action == 0:#left
        direction = -20
    else:#right
        direction = 20

    p.setJointMotorControl2(0 , 0 , p.TORQUE_CONTROL, force = direction)
    p.stepSimulation()
    self.state = np.array(p.getJointStates(0 , [0 ,1] ))[:,:2].flatten()
    playing = True
    reward =  1
    return reward , playing

env = PyBulletCartPoleEnv()

class Actor_critic(tf.keras.Model):
  def __init__( self , num_actions , num_hidden_units ):
    super().__init__()

    self.shared_1 = layers.Dense( num_hidden_units , activation ='relu')
    self.actor = layers.Dense(num_actions)
    self.critic = layers.Dense(1)

  def call( self , input_obs ):
    x = self.shared_1(input_obs)
    return self.actor(x) , self.critic(x)

model = Actor_critic( num_actions = 2 , num_hidden_units = 256 )

def step_episode( env , model ):
  env.reset()
  action_probs_list = []
  tar_vals = []
  pred_vals = []
  playing = True
  while playing == True:
    state = tf.expand_dims( env.state.tolist() , 0 )
    #run model to get action logits and value
    action_logits , pred_value = model(state)
    pred_vals.append(pred_value[0][0])
    #categorical probabilistic action idx selection
    selected_action_idx = tf.random.categorical( action_logits , 1 )[ 0 , 0 ]
    reward , playing = env.step(selected_action_idx)
    next_state = tf.expand_dims( env.state.tolist() , 0 )
    _ , next_state_value = model(next_state)
    target_val = reward + next_state_value[0][0] * .9
    tar_vals.append(target_val)
    #normalized probs
    action_probs = tf.nn.softmax(action_logits)
    probability_of_taking_selected_action = action_probs[0 , selected_action_idx]
    action_probs_list.append(probability_of_taking_selected_action)
  return action_probs_list , tar_vals , pred_vals

#this returns all lists flat!!!!!
def episode_batch(env , model , num_episodes ):
    action_probs_list = []
    pred_list = []
    tar_list = []
    for episode in range( num_episodes ):
        action_probs , tar , pred = step_episode( env , model )
        action_probs_list.append( action_probs )
        pred_list.append( pred )
        tar_list.append( tar )
    return action_probs_list  , tar_list , pred_list

def actor_loss_fn( action_probs ,  tar_vals ):
  # log of pi (a | s)
  action_log_probs = tf.math.log(action_probs)
  actor_loss =  - tf.math.reduce_sum(action_log_probs * tar_vals ) / ( len( tar_vals ))
  return actor_loss

huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

def critic_loss_fn( predictions ,  targets ):
  l = huber_loss( targets , predictions )
  return l

def actor_critic_loss(action_probs , predictions , targets ):
  actor_loss_val = actor_loss_fn(action_probs , targets )
  critic_loss_val = critic_loss_fn( predictions , targets )
  return actor_loss_val + critic_loss_val

optimizer= tf.keras.optimizers.Adam(learning_rate=.0001 )

average_len = []
episodes_per_batch = 1
for episode in range(20000):
  with tf.GradientTape() as tape:
    action_probs , tar_vals , pred_vals = episode_batch( env , model , episodes_per_batch )
    loss = actor_critic_loss( action_probs , pred_vals , tar_vals )
    grads = tape.gradient(loss , model.trainable_variables )
    optimizer.apply_gradients(zip(grads , model.trainable_variables))
    average_len.append(len(action_probs[0])/episodes_per_batch)#adds episode average reward
    if episode % 10 == 0:
      print(np.mean(average_len[-50:]) )

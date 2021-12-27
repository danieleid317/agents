import numpy as np
import random
import statistics
import tensorflow as tf

from matplotlib import pyplot as plt
from tensorflow.keras import layers

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

class Actor_critic_gru(tf.keras.Model):
  def __init__( self , num_actions , num_hidden_units ):
    super().__init__()

    self.shared_1 = layers.Dense( num_hidden_units , activation ='relu')
    self.actor = layers.Dense(num_actions)
    self.critic = layers.Dense(1)
    self.gru = layers.GRU(num_hidden_units)

  def call( self , input_obs ):
    x = self.gru(input_obs)
    return self.actor(x) , self.critic(x)

model = Actor_critic_gru( num_actions = 2 , num_hidden_units = 100)

def step_episode( env , model ):
  env.reset()
  action_probs_list = []
  tar_vals = []
  pred_vals = []
  advantages = []
  obs_traj = []
  playing = True
  obs_traj.append(env.state)

  while playing == True:
    action_logits , pred_value = model(tf.expand_dims( obs_traj , 0 ))
    pred_vals.append(pred_value[0][0])

    #categorical probabilistic action idx selection
    selected_action_idx = tf.random.categorical( action_logits , 1 )[ 0 , 0 ]
    #step env
    reward , playing = env.step(selected_action_idx)
    #new obs -> traj
    obs_traj.append(env.state)

    _ , next_value = model(tf.expand_dims( obs_traj , 0 )) # run model on next state
    target_val = reward + ( next_value[0][0] * .95  )
    tar_vals.append(target_val)
    advantages.append( target_val - pred_value )

  return action_probs_list , tar_vals , pred_vals , advantages


def actor_loss_fn( action_probs ,  advantages ):
  # log of pi (a | s)
  action_log_probs = tf.math.log(action_probs)
  actor_loss =  - tf.math.reduce_sum(action_log_probs * advantages ) / ( len( advantages ))
  return actor_loss

huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
def critic_loss_fn( predictions ,  targets ):
  l = huber_loss( targets , predictions )
  return l

def actor_critic_loss(action_probs , predictions , targets , advantages ):
  actor_loss_val = actor_loss_fn(action_probs , advantages )
  critic_loss_val = critic_loss_fn( predictions , targets )
  return actor_loss_val + critic_loss_val

optimizer= tf.keras.optimizers.Adam(learning_rate=.0001 )

average_len = []

for episode in range(20000):
  with tf.GradientTape() as tape:
    action_probs , tar_vals , pred_vals ,advantages = step_episode( env , model )
    loss = actor_critic_loss( action_probs , pred_vals , tar_vals , advantages )
    grads = tape.gradient(loss , model.trainable_variables )
    optimizer.apply_gradients(zip(grads , model.trainable_variables))
    average_len.append(len(action_probs[0])/episodes_per_batch)#adds episode average reward
    if episode % 10 == 0:
      print(np.mean(average_len[-50:]) )

'''
Below are two implementations of the cartpole environment built using
PyBullet.
One environment is a discrete implementation where the actions are left/right,
while the other env takes continuous actions that are applied as torques to the cart.

Also two implementations of a 2-D gridworld.  There are gridworlds with both single
and double targets.  The obs/state provided do not indicate whether or not any target
has been reached.
'''

import numpy as np
import random
import statistics

import pybullet as p
import pybullet_data

import time

class DiscretePyBulletCartPoleEnv():
    #discrete implementation of cartpole environment
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

class ContinuousPyBulletCartPoleEnv():
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
    if (self.step_count >= 600) :
      self.reset()
      reward =   1
      done = True
      return reward , done

    if (abs(self.state[0]) > 3) :#moves too far outside horizontaly on cart
      self.reset()
      done = True
      reward = 0
      return reward , done

    if abs(self.state[2]) >= .2 :# not within angle
      self.reset()
      done = True
      reward = 0
      return reward , done

    p.setJointMotorControl2(0 , 0 , p.TORQUE_CONTROL, force = action)
    p.stepSimulation()

    self.state = np.array(p.getJointStates(0 , [0 ,1] ))[:,:2].flatten()
    done = False
    reward =  1
    return reward , done

class OneTargetGridworld():
  def __init__(self):
    self.state = self.init_state()
    self.step_count = 0
    self.first = 0
  def init_state(self):
    xs = random.sample(range(0,7),5)
    ys = random.sample(range(0,7),5)
    piece1 = [xs[0],ys[0]]
    piece2 = [xs[1],ys[1]]

    playerpos = [xs[4],ys[4]]
    positions = list(np.array([ playerpos , piece1 , piece2 ]).flatten())

    return positions

  def reset(self):
    self.state = self.init_state()
    self.step_count = 0

  def step(self, action):
    self.step_count += 1
    if (self.step_count >= 25) : # loss maxsteps
      self.reset()
      reward =   - 10
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



class TwoTargetGridworld():
  def __init__(self):
    self.state = self.init_state()
    self.step_count = 0
    self.check = 0

  def init_state(self):
    xs = random.sample(range(0,5),5)
    ys = random.sample(range(0,5),5)
    piece1 = [xs[0],ys[0]]
    piece2 = [xs[1],ys[1]]
    playerpos = [xs[4],ys[4]]
    positions = list(np.array([ playerpos , piece1 , piece2 ]).flatten())
    return positions

  def reset(self):
    self.state = self.init_state()
    self.step_count = 0
    self.check = 0

  def step(self, action):
    self.step_count += 1
    if (self.step_count >= 70) : # loss maxsteps
      self.reset()
      reward =   - 10
      playing = False
      return reward , playing

    if ( self.state[0] == self.state[2] ) and ( self.state[1] == self.state[3] ) and (self.check == 0) : # first checkpoint
      self.check = 1
      playing = True
      reward = 100
      return reward , playing

    if ( self.state[0] == self.state[4] ) and ( self.state[1] == self.state[5] ) and (self.check == 1 ) : # second and final success condition
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



def decode_action(action):
  finmove = []
  if action == 0 :
      finmove = [ -.5 , 0 , 0 ]
  if action == 1 :
      finmove = [ 0 , -.5 , 0 ]
  if action == 2 :
      finmove = [ 0 , 0 , -.5 ]
  if action == 3 :
      finmove = [ .5 , 0 , 0 ]
  if action == 4 :
      finmove = [ 0 , .5 , 0 ]
  if action == 5 :
      finmove = [ 0 , 0 , .5 ]
  return finmove

class PyBulletGoalFinder():
  def __init__(self):
    self._state = self.init_state()
    self.step_count = 0

  def init_state(self):
    p.connect(p.GUI)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0 , 0 , 0)
    locations = np.random.randint(0 , 5 , size = (2 , 3))
    p.loadURDF("plane.urdf" , [0,0,0] , [ 0 , 0 , 0 , 1 ])
    p.loadURDF("cube.urdf" , locations[0] , [ 0 , 0 , 0 , 1 ])
    p.loadURDF("soccerball.urdf" , locations[1] , [ 0 , 0 , 0 , 1 ])
    playerPos = p.getBasePositionAndOrientation(1)[0]
    targetPos = p.getBasePositionAndOrientation(2)[0]
    obs = np.array([ playerPos , targetPos]).flatten()
    return obs

  def reset(self):
    p.disconnect()
    self.state = self.init_state()
    self.step_count = 0

  def step(self, action):
    self.step_count+=1
    time.sleep(.02)

    touching = p.getContactPoints(1,2)
    if touching :#success condition
      reward = 10
      playing = False
      return reward , playing

    if (self.step_count >= 100) :# maxsteps fail
      playing = False
      reward = - 10
      return reward , playing

    playerPos = p.getBasePositionAndOrientation(1)[0]
    targetPos = p.getBasePositionAndOrientation(2)[0]
    obs = np.array([ playerPos , targetPos]).flatten()
    newPos = np.add(np.array(playerPos) , decode_action(action))
    p.resetBasePositionAndOrientation(1 , newPos ,[ 0 , 0 , 0 , 1 ])
    p.stepSimulation()
    self.state = obs
    reward = -1
    playing = True
    reward = -sum(abs(np.array(playerPos) - np.array(targetPos))) #pseudo value reward of distance to target
    return reward , playing

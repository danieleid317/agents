import numpy as np
import pybullet as p
import pybullet_data
import time
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

'''
RNN Continuous Implementation
'''


import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from envs import PyBulletCartPoleEnv
import statistics
env = PyBulletCartPoleEnv()

class RNN(nn.Module):
    def __init__(self, input_size , hidden_size, output_size , lr):
        super(RNN, self).__init__()
        self.lr = lr
        self.hidden_size = hidden_size
        self.hidden_state_layer = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_layer = nn.Linear(input_size + hidden_size, output_size)

        self.optimizer = optim.Adam(self.parameters() , lr = self.lr )

    def forward(self, input, hidden):
        x = T.tensor(input , dtype = T.float)
        combined = T.cat((x, hidden), 1)
        hidden = self.hidden_state_layer(combined)
        output = self.output_layer(combined)
        return output, hidden

    def init_hidden(self):
        return T.zeros(1, self.hidden_size)

input_size = 4
n_outputs = 2

class Agent(object):
  def __init__( self , actor_lr , critic_lr , input_dims , gamma = 0.9 , n_actions = 2 , layer1_size = 64 , layer2_size = 64 , n_outputs = 2 ):
    self.gamma = gamma
    self.log_probs = []
    self.advantages = []
    self.estimate = None
    self.ground_truth = None

    self.actor = RNN( input_size , hidden_size=100 , output_size = n_outputs , lr = actor_lr )
    self.critic = RNN(input_size , hidden_size=100 , output_size=1 , lr = critic_lr)
    self.actor_hidden = None
    self.critic_hidden = None

  def choose_action( self , obs_traj ):
    agent.actor_hidden = agent.actor.init_hidden()
    x = T.unsqueeze(T.tensor(obs_traj),1)
    for i in range(x.size()[0]):
        mu_sigma , self.actor_hidden = self.actor.forward(x[i], self.actor_hidden)
    sigma = T.abs( mu_sigma[0][1] )
    mu = T.tanh(mu_sigma[0][0])
    distribution = T.distributions.Normal ( mu , sigma )
    torque = distribution.sample(sample_shape = (1,))
    self.log_probs = distribution.log_prob(torque)
    action = T.tanh(torque)
    return action.item()

  def log_step(self , reward , obs_traj_ , prime_state , done):
    x = T.unsqueeze(T.tensor(obs_traj_),1)
    agent.critic_hidden = agent.critic.init_hidden()
    for i in range(x.size()[0]):
        critic_value, self.critic_hidden = self.critic.forward(x[i], self.critic_hidden)

    self.estimate = critic_value
    reward = T.tensor( reward , dtype = T.float )

    obs_traj_.append(prime_state)
    next_state_traj = T.unsqueeze(T.tensor(obs_traj_),1)
    agent.critic_hidden = agent.critic.init_hidden()
    with T.no_grad():
      for i in range(next_state_traj.size()[0]):
        critic_value_ , self.critic_hidden = self.critic.forward( next_state_traj[i] , self.critic_hidden)
    self.ground_truth = reward + critic_value_
    advantage = reward  + self.gamma*critic_value_*(1-int(done)) - critic_value
    self.advantages =  advantage

  def learn( self , ):
    actor_loss = -self.log_probs * self.advantages
    critic_loss = ( self.estimate - self.ground_truth ) ** 2
    ( actor_loss + critic_loss ).backward()
    self.actor.optimizer.step()
    self.critic.optimizer.step()
    self.actor.optimizer.zero_grad()
    self.critic.optimizer.zero_grad()

agent = Agent( actor_lr = 0.000001 , critic_lr = .000001 , input_dims = 4 , gamma = .9 , layer1_size = 256 , layer2_size = 256 , n_outputs = 2 )

lens = []
num_episodes = 50
for i in range(num_episodes):
  obs_traj = []
  done = False
  env.reset()
  while not done :
    obs_traj.append( env.state.tolist() )
    action = agent.choose_action(obs_traj)
    reward , done = env.step(action)
    agent.log_step( reward , obs_traj , env.state.tolist() , done)
    agent.learn()
  lens.append(len(obs_traj))
  print(statistics.mean(lens[-10:]))

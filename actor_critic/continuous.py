import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from envs import PyBulletCartPoleEnv

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
    self.obs_traj = []


    self.actor = RNN( input_size , hidden_size=100 , output_size = n_outputs , lr = actor_lr )
    self.critic = RNN(input_size , hidden_size=100 , output_size=1 , lr = critic_lr)
    self.actor_hidden = self.actor.init_hidden()
    self.critic_hidden = self.critic.init_hidden()

  def choose_action( self , observation ):
    self.obs_traj.append( observation )
    mu_sigma , hidden = self.actor.forward( self.obs_traj , self.actor_hidden )
    self.actor_hidden = hidden
    sigma = T.abs( mu_sigma[0][1] )
    distribution = T.distributions.Normal ( mu_sigma[0][0] , mu_sigma[0][1] )
    torque = distribution.sample(sample_shape = (1,))
    self.log_probs = distribution.log_prob(torque)
    action = T.tanh(torque)
    return action.item()

  def log_step(self , reward , new_state , done):
    critic_value , hidden = self.critic.forward(self.obs_traj , self.critic_hidden)
    self.critic_hidden = hidden
    #self.estimates.append( critic_value )
    reward = T.tensor( reward , dtype = T.float )
    self.obs_traj.append( new_state )
    with T.no_grad():
      critic_value_ , _ = self.critic.forward(self.obs_traj , self.critic_hidden)
    self.obs_traj.drop( new_state )
    #self.ground_truths = reward + critic_value_
    advantage = reward + self.gamma*critic_value_*(1-int(done)) - critic_value
    self.advantages =  advantage

  def learn( self , ):
    actor_loss = -self.log_probs * self.advantages
    critic_loss = T.reduce_difference( self.estimates - self.ground_truths ) ** 2
    ( actor_loss + critic_loss ).backward()
    self.actor.optimizer.step()
    self.critic.optimizer.step()
    self.actor.optimizer.zero_grad()
    self.critic.optimizer.zero_grad()

agent = Agent( actor_lr = 0.001 , critic_lr = .001 , input_dims = 4 , gamma = .9 , layer1_size = 256 , layer2_size = 256 , n_outputs = 2 )

num_episodes = 50000
for i in range(num_episodes):

  done = False
  score = 0
  env.reset()
  observation = env.state
  while not done :
    action = agent.choose_action(env.state)
    reward , done = env.step(action)
    observation_ = env.state
    agent.log_step( reward , observation_ , done)
    score += reward
    agent.learn()
  print(score)

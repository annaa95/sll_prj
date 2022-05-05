import os

import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from noise_generator import OUActionNoise
#from replayBuffer import ReplayBuffer
from pathlib import Path


class CriticNetwork(nn.Module):
    def __init__(self,
                 lr,
                 input_dims,
                 fc1_dims,
                 fc2_dims,
                 fc3_dims,
                 n_actions,
                 name,
                 chkpt_dir='/home/anna/catkin_ws/src/sll_extern_roscontroller/scripts/supervisorManager/models/PhilTabor/saved/ddpg/'):
        super(CriticNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims

        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_ddpg')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.action_value = nn.Linear(self.n_actions, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims + fc2_dims, fc3_dims)
        self.q = nn.Linear(fc3_dims, 1)

        self.initialization()

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def initialization(self):
        nn.init.xavier_uniform_(self.fc1.weight,
                                gain=nn.init.calculate_gain('leaky_relu'))

        nn.init.xavier_uniform_(self.fc2.weight,
                                gain=nn.init.calculate_gain('leaky_relu'))

        nn.init.xavier_uniform_(self.action_value.weight,
                                gain=nn.init.calculate_gain('leaky_relu'))

        nn.init.xavier_uniform_(self.fc3.weight,
                                gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = F.leaky_relu(state_value)
        # state_value = self.bn1(state_value)

        state_value = self.fc2(state_value)
        state_value = F.leaky_relu(state_value)
        # state_value = self.bn2(state_value)

        action_value = self.action_value(action)
        action_value = F.leaky_relu(action_value)

        state_action_value = T.cat((action_value, state_value), dim=1)
        state_action_value = self.fc3(state_action_value)
        state_action_value = F.relu(state_action_value)

        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        print("...saving checkpoint....")
        print(self.checkpoint_file)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("..loading checkpoint...")
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self,
                 lr,
                 input_dims,
                 fc1_dims,
                 fc2_dims,
                 fc3_dims,
                 n_actions,
                 name,
                 chkpt_dir='/home/anna/catkin_ws/src/sll_extern_roscontroller/scripts/supervisorManager/models/PhilTabor/saved/ddpg/'):
        super(ActorNetwork, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims

        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name + "_ddpg")

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        # self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.bn3 = nn.LayerNorm(self.fc3_dims)

        self.mu = nn.Linear(self.fc3_dims, self.n_actions)

        self.initialization()

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def initialization(self):
        nn.init.xavier_uniform_(self.fc1.weight,
                                gain=nn.init.calculate_gain('leaky_relu'))

        nn.init.xavier_uniform_(self.fc2.weight,
                                gain=nn.init.calculate_gain('leaky_relu'))

        nn.init.xavier_uniform_(self.fc3.weight,
                                gain=nn.init.calculate_gain('leaky_relu'))

        nn.init.xavier_uniform_(self.mu.weight,
                                gain=nn.init.calculate_gain('tanh'))

    def forward(self, state):
        x = self.fc1(state)
        x = F.leaky_relu(x)
        # x = self.bn1(x)

        x = self.fc2(x)
        x = F.leaky_relu(x)
        # x = self.bn2(x)

        x = self.fc3(x)
        x = F.leaky_relu(x)
        # x = self.bn3(x)

        x = T.sigmoid(self.mu(x))

        return x

    def save_checkpoint(self):
        print("...saving checkpoint....")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("..loading checkpoint...")
        self.load_state_dict(T.load(self.checkpoint_file))


class DDPG(object):
    def __init__(self,
                 lr_actor,
                 lr_critic,
                 input_dims,
                 tau,
                 env,                 
                 n_actions,
                  gamma=0.99,
                 max_size=1000000,
                 layer1_size=400,
                 layer2_size=300,
                 layer3_size=200,
                 batch_size=64,
                 load_models=False,
                 save_dir='/home/anna/catkin_ws/src/sll_extern_roscontroller/scripts/supervisorManager/models/PhilTabor/saved/ddpg/'):
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        if load_models:
            self.load_models(lr_critic, lr_actor, input_dims, layer1_size,
                             layer2_size, layer3_size, n_actions, save_dir)
        else:
            self.init_models(lr_critic, lr_actor, input_dims, layer1_size,
                             layer2_size, layer3_size, n_actions, save_dir)

        self.noise = OUActionNoise(mu=np.zeros(n_actions),
                                   dt=1e-2
                                   # sigma=0.3,
                                   # theta=0.15,
                                   )

        self.update_network_parameters(tau=1)

    def choose_action_train(self, observation):
        if observation is not None:
            self.actor.eval()
            observation = T.tensor(observation,
                                   dtype=T.float).to(self.actor.device)
            mu = self.actor(observation).to(self.actor.device)
            noise = T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
            # print("Noise {}, Mu {}".format(noise, mu))
            mu_prime = mu + noise
            self.actor.train()
            return mu_prime.cpu().detach().numpy()
        return np.zeros((2, ))

    def choose_action_test(self, observation):
        if observation is not None:
            self.actor.eval()
            observation = T.tensor(observation,
                                   dtype=T.float).to(self.actor.device)
            mu = self.target_actor(observation).to(self.target_actor.device)

            return mu.cpu().detach().numpy()
        return np.zeros((2, ))

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = \
                                                 self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        done = T.tensor(done).to(self.critic.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        state = T.tensor(state, dtype=T.float).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_actions = self.target_actor.forward(new_state)
        critic_value_ = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state, action)

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma * critic_value_[j] * done[j])

        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)

        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def work(self):
        self.target_actor.eval()
        self.target_critic.eval()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()

        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)

        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() +\
                                     (1-tau)*target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() +\
                                     (1-tau)*target_actor_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

    def init_models(self, lr_critic, lr_actor, input_dims, layer1_size,
                    layer2_size, layer3_size, n_actions, save_dir):
        self.actor = ActorNetwork(lr_actor,
                                  input_dims,
                                  layer1_size,
                                  layer2_size,
                                  layer3_size,
                                  n_actions=n_actions,
                                  name="Actor",
                                  chkpt_dir=save_dir)

        self.target_actor = ActorNetwork(lr_actor,
                                         input_dims,
                                         layer1_size,
                                         layer2_size,
                                         layer3_size,
                                         n_actions=n_actions,
                                         name="TargetActor",
                                         chkpt_dir=save_dir)

        self.critic = CriticNetwork(lr_critic,
                                    input_dims,
                                    layer1_size,
                                    layer2_size,
                                    layer3_size,
                                    n_actions=n_actions,
                                    name="Critic",
                                    chkpt_dir=save_dir)

        self.target_critic = CriticNetwork(lr_critic,
                                           input_dims,
                                           layer1_size,
                                           layer2_size,
                                           layer3_size,
                                           n_actions=n_actions,
                                           name="TargetCritic",
                                           chkpt_dir=save_dir)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self, lr_critic, lr_actor, input_dims, layer1_size,
                    layer2_size, layer3_size, n_actions, load_dir):

        self.init_models(lr_critic, lr_actor, input_dims, layer1_size,
                         layer2_size, layer3_size, n_actions, load_dir)

        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        # the call method allow us to do in our code smt like:
        #       noise = OUActionNoise()
        #       ourNoise = noise()   
        x = self.x_prev + self.theta*(self.mu-self.x_prev)*self.dt+self.sigma*np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0 #counter, if we reach max_size we have to reward the future with 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        # store the tuple SARS' in the Replay Buffer
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = 1 - done # in order not to count terminal states
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        # in order not to sample zeros when the buffer is still partially empty 
        # we set max_mem to the minimum between mem_cntr and mem_size
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        reward = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, reward, new_states, terminal        
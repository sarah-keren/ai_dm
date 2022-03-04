"""

Policy gradient implementaiton using PyTorch

https://www.youtube.com/watch?v=GOBvUA9lK1Q (sorry)

Details of below:
Two fully connected hidden layers, default 256 neurons each 
Relu activation functions, softmax final activation function
Update according to log gradient
With standardized updates

TODO: add option to use cnn instead of fc (e.g. pixel env)
TODO: check if working
"""

import torch as T 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import numpy as np 

class PolicyNetwork(nn.Module): # The network itself is separate from the agent
    def __init__(self, learning_rate, input_dims, fc1_dims, fc2_dims, num_actions, mapping_fn=None):
        super(PolicyNetwork, self).__init__()

        self.input_dims = input_dims 
        self.learning_rate = learning_rate
        self.fc1_dims = fc1_dims 
        self.fc2_dims = fc2_dims
        self.num_actions = num_actions 
        self.mapping_fn = mapping_fn

        self.fc1 = nn.Linear(self.input_dims, self.fc2_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.num_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    def forward(self, observation):
        if self.mapping_fn:
            observation = self.mapping_fn(observation) 

        state = T.Tensor(observation).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    
class DPGAgent(object):
    def __init__(self, learning_rate, input_dims, num_actions, gamma=0.99, l1_size=256, l2_size=256, mapping_fn=None, action_return_format=None):
        self.gamma = gamma 
        self.reward_mem = [] 
        self.action_mem = [] 
        self.num_actions = num_actions
        self.gamma = gamma
        self.policy = PolicyNetwork(learning_rate, input_dims, l1_size, l2_size, num_actions, mapping_fn)
        self.action_return_format = action_return_format


    def int_to_vector(self, action):
        """ Turns integer action into one hot vector """
        vec = np.zeros(self.num_actions)
        vec[action] = 1 
        return vec 
    
    def learn(self):
        """ Calculate rewards for the episode, compute the gradient of the loss, update optimizer"""
        self.policy.optimizer.zero_grad()
        G = np.zeros_like(self.reward_mem, dtype=np.float64)

        for t in range(len(self.reward_mem)):
            G_sum = 0 
            discount = 1

            for k in range(t, len(self.reward_mem)):
                G_sum += self.reward_mem[k] * discount
                discount *= self.gamma 
            G[t] = G_sum 
        
        # standardize updates
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        G = (G-mean)/std 

        G = T.tensor(G, dtype=T.float).to(self.policy.device)
        loss = 0

        for g, logprob in zip(G, self.action_mem):
            loss += -g * logprob 
        
        loss.backward()

        self.policy.optimizer.step()

    """ Training Callbacks """
    def action_callback(self, observation):
        probs = F.softmax(self.policy.forward(observation))
        action_probs = T.distributions.Categorical(probs)
        action = action_probs.sample()

        log_probs = action_probs.log_prob(action)
        self.action_mem.append(log_probs)

        returned_action = action.item()
        if self.action_return_format == 'vector':
            returned_action = self.int_to_vector(returned_action)

        #return action.item()
        return returned_action

    def experience_callback(self, obs, action, new_obs, reward, done):
        self.reward_mem.append(reward)

    def episode_callback(self):
        """ Reset at the end of an episode"""
        self.learn() 
        self.reward_mem = [] 
        self.action_mem = [] 
    
    """ Evaluation Callbacks """
    def policy_callback(self, observation):
        probs = F.softmax(self.policy.forward(observation))
        action_probs = T.distributions.Categorical(probs)
        action = action_probs.sample()
        
        return action.item() 
    
    def reset(self):
        return


def test_continuous_single_agent():

    import gym
    env = gym.make('CartPole-v0')
    num_actions = env.action_space.n
    num_states = env.observation_space.shape[0]
    theta = np.random.rand(num_states, num_actions)
    learning_rate = 0.01

    dpg_agent = DPGAgent(learning_rate, num_states, num_actions, gamma=0.99, l1_size=256, l2_size=256, mapping_fn=None)

    import train
    train.train(env=env, is_env_multiagent=False, agents=[dpg_agent], max_episode_len=10000, num_episodes=10000,
                method='train', display=False, save_rate=10, agents_save_path="", train_result_path="")


def test_continuous_multi_agent():
    import train
    #A simple multi-agent particle world with a continuous observation and discrete action space, along with some basic simulated physics.
    # Used in the paper [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf)
    # the code for the environment can be found in https://github.com/openai/multiagent-particle-envs
    import sys
    sys.path.append("../particle_env")
    #import particle_env
    from make_env import make_env
    env = make_env('simple_speaker_listener')


    # init agents
    learning_rate = 0.01

    dpg_agent_speaker = DPGAgent(learning_rate, input_dims=3, num_actions=3, gamma=0.99, l1_size=256, l2_size=256, mapping_fn=None, action_return_format="vector")
    dpg_agent_listener = DPGAgent(learning_rate, input_dims=11, num_actions=5, gamma=0.99, l1_size=256, l2_size=256,mapping_fn=None, action_return_format="vector")

    # run train
    train.train(env=env, is_env_multiagent=True, agents=[dpg_agent_speaker, dpg_agent_listener], max_episode_len=25, num_episodes=10000,
                method='train', display=False, save_rate=10, agents_save_path="", train_result_path="")


if __name__ == "__main__":
    #test_continuous_single_agent()
    test_continuous_multi_agent()
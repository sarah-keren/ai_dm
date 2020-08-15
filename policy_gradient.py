import numpy as np 
""" 
Policy Gradient Agent 

Softmax policy gradient implementation

Args:
theta: weights initialization
gamma: discount parameter
alpha: scaling parameter
num_actions: int (must be same as first dimension of dim)
mapping_fn(state) => state: optional function to transform an observatiohn

Implements:
action_callback(state): returns action integer
reward_callback(reward): 
reset: callback at end of each episode (updates weights)

Implemented (hopefully) according to Sutton 2018 Ch. 13 REINFORCE: Monte Carlo Policy Gradient (episodic)
update according to REINFORCE update (13.8)

Preferences are linear in features (13.3)
Actions and gradients derived according to softmax distribution (13.2)
"""

class PolicyGradientAgent(object):
    """ Generic Policy Gradient Implementation """
    def __init__(self, num_actions, theta, alpha=0.00025, gamma=0.9, mapping_fn=None):
        """ Parameters """
        self.theta = theta
        self.alpha = alpha 
        self.gamma = gamma

        """ Record keeping / episode """
        self.grads = []
        self.rewards = []

        self.mapping_fn = mapping_fn
        self.num_actions = num_actions

    def int_to_vector(self, action):
        """ Turns integer action into one hot vector """
        vec = np.zeros(self.num_actions)
        vec[action] = 1 
        return vec 
    
    def softmax(self, state):
        """ softmax(state * weights) """
        z = state.dot(self.theta)
        exp = np.exp(z)
        return exp/np.sum(exp)

    def policy(self, state):
        """ Returns agent policy given state """
        probs = self.softmax(state)
        return probs
    
    def softmax_gradient(self, softmax):
        """ Derivative of the softmax w.r.t. theta """
        s = softmax.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)
    
    def compute_gradient(self, probs, state, action):
        """ Computes the gradient of log(softmax) for a state and action """
        dsoftmax = self.softmax_gradient(probs)[action, :]
        dlog = dsoftmax / probs[0, action]
        grad = state.T.dot(dlog[None,:])
        
        self.grads.append(grad)
        return 

    def update_weights(self):
        """ Update weights """
        for i in range(len(self.grads)):
            present_val_of_rewards = sum([r * (self.gamma ** r) \
                        for t, r in enumerate(self.rewards[i:])])
            self.theta += self.alpha * self.grads[i] * present_val_of_rewards 
        return 

    """ Training Callbacks """
    def action_callback(self, state):
        """ Take an action according to policy. Save gradient."""
        if self.mapping_fn:
            state = self.mapping_fn(state)

        state = state[None,:]
        probs = self.policy(state)

        action = np.random.choice(self.num_actions, p=probs[0])
        
        self.compute_gradient(probs, state, action)
        
        return action

    def experience_callback(self, obs, action, new_obs, reward, done):
        """ Receive rewards """
        self.rewards.append(reward)
    
    def episode_callback(self):
        """ Update weights, reset records for new episodes"""
        self.update_weights()

        self.grads = []
        self.rewards = []

    """ Evaluation callbacks """
    def policy_callback(self, state):
        """ Take an action according to policy."""
        if self.mapping_fn:
            state = state.mapping_fn(state)
        
        state = state[None,:]
        probs = self.policy(state) 
        action = np.random.choice(self.num_actions, p=probs[0])

        return action 
    
    def reset(self):
        """ Reset records for new episodes """
        return 



def test_discrete():

    import gym
    env = gym.make("Taxi-v3").env
    env.reset()
    env.render()

    num_actions = env.action_space.n
    num_states = env.observation_space.n
    np.zeros([env.observation_space.n, env.action_space.n])
    theta = np.random.rand(num_actions, num_states)
    pg_agent = PolicyGradientAgent(num_actions, theta, alpha=0.00025, gamma=0.9, mapping_fn=None)


def test_continuous():
    import gym
    env = gym.make('CartPole-v0')
    num_actions = env.action_space.n
    num_states = env.observation_space.shape[0]
    theta = np.random.rand(num_actions, num_states)

    pg_agent = PolicyGradientAgent(num_actions, theta, alpha=0.00025, gamma=0.9, mapping_fn=None)
    #action_sets = [env.get_action_set(agents, obs, method)]

    import train
    train.train(env=env, agents=[pg_agent], max_episode_len=10000, num_episodes=10000, method='train', display=False, save_rate=10, save_path="", train_result_path="")

if __name__ == "__main__":
    #test_discrete()
    test_continuous()

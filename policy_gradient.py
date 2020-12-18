from learning_agent import RL_Agent
import numpy as np
import sys

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


class PolicyGradientAgent(RL_Agent):
    """ Generic Policy Gradient Implementation """

    def __init__(self, action_return_format=None, **args):
        super().__init__(**args)
        """ Record keeping / episode """
        self.grads = []
        self.rewards = []

        # environment may want vector instead. Can define on environment but it has some consequences depending on which
        # environment you are working with. This may be a bit more general, but should think about whether it is better on the
        # environment.
        self.action_return_format = action_return_format

    def int_to_vector(self, action):
        """ Turns integer action into one hot vector """
        vec = np.zeros(self.num_actions)
        vec[action] = 1
        return vec

    def softmax(self, state):
        """ softmax(state * weights) """
        z = state.dot(self.theta)
        exp = np.exp(z - np.max(z))
        return exp / np.sum(exp)

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
        dsoftmax_comp = self.softmax_gradient(probs)
        dsoftmax = dsoftmax_comp[action, :]
        dlog = dsoftmax / probs[0, action]
        grad = state.T.dot(dlog[None, :])

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
        """ Take an action according to policy. Save gradient. Should use transform_fn if it exist."""
        if self.mapping_fn:
            state = self.mapping_fn(state, self.num_actions)

        state = state[None, :]
        probs = self.policy(state)

        action = np.random.choice(self.num_actions, p=probs[0])

        self.compute_gradient(probs, state, action)
        if self.action_return_format == 'vector':
            return self.int_to_vector(action)

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

        state = state[None, :]
        probs = self.policy(state)
        action = np.random.choice(self.num_actions, p=probs[0])

        if self.action_return_format == 'vector':
            return self.int_to_vector(action)

        return action

    def reset(self):
        """ Reset records for new episodes """
        return


def test_discrete():
    import gym
    env = gym.make("Taxi-v3").env  # policy gradient not well suited to taxi env. should work but may take long time
    env.reset()
    env.render()

    num_actions = env.action_space.n
    num_states = env.observation_space.n
    theta = np.random.rand(num_states, num_actions)

    pg_agent = PolicyGradientAgent(num_actions=num_actions, theta=theta, alpha=0.025, gamma=0.9,
                                   mapping_fn=lambda x: np.squeeze(np.eye(500)[np.array(x).reshape(-1)]) / 500)

    import train_and_evaluate
    train_and_evaluate.train(env=env, is_env_multiagent=False, agents=[pg_agent], max_episode_len=10000, num_episodes=1000,
                display=False, save_rate=1, agents_save_path="", train_result_path="")


def test_continuous_single_agent():
    import gym
    env = gym.make('CartPole-v0')
    num_actions = env.action_space.n
    num_states_vars = env.observation_space.shape[0]
    theta = np.random.rand(num_states_vars, num_actions)
    pg_agent = PolicyGradientAgent(num_actions=num_actions, theta=theta, alpha=0.025, gamma=0.9, mapping_fn=None)

    import train_and_evaluate
    train_and_evaluate.train(env=env, is_env_multiagent=False, agents=[pg_agent], max_episode_len=100,
                             num_episodes=1000, display=False, save_rate=10, agents_save_path="", train_result_path="")

    train_and_evaluate.evaluate(env=env, is_env_multiagent=False, agents=[pg_agent], max_episode_len=100,
                                num_episodes=100, display=True, save_rate=10, agents_save_path="", train_result_path="")


def test_continuous_multi_agent():
    import train_and_evaluate
    # A simple multi-agent particle world with a continuous observation and discrete action space, along with some basic simulated physics.
    # Used in the paper [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf)
    # the code for the environment can be found in https://github.com/openai/multiagent-particle-envs
    sys.path.append("../particle_env")
    # import particle_env
    from make_env import make_env
    env = make_env('simple_speaker_listener')
    # init agents
    agent1 = PolicyGradientAgent(num_actions=3, theta=np.random.rand(3, 3), action_return_format="vector")  # speaker
    agent2 = PolicyGradientAgent(num_actions=5, theta=np.random.rand(11, 5), action_return_format="vector")  # listener

    # run train
    train_and_evaluate.train(env=env, is_env_multiagent=True, agents=[agent1, agent2], max_episode_len=25,
                             num_episodes=10000, display=True, save_rate=10, agents_save_path="", train_result_path="")
    # evaluate performance
    train_and_evaluate.evaluate(env=env, is_env_multiagent=True, agents=[agent1, agent2], max_episode_len=25,
                                num_episodes=100, display=True, save_rate=10, agents_save_path="", train_result_path="")


if __name__ == "__main__":
    test_discrete()
    # test_continuous_single_agent()
    # test_continuous_multi_agent()

from abc import ABC
import random

import agent_utils
from learning_agent import RL_Agent

MAX_EPISODE_LEN = 100


class MCAgents(RL_Agent, ABC):
    def __init__(self, env, epsilon=0.2, discount=0.9, **args):
        super().__init__(**args)
        self.epsilon = epsilon
        self.discount = discount
        self.env = env
        self.q_values = {}
        self.episode = []
        self.generate_episode()
        self.t = 0
        self.total_return = {}

    def get_q_value(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we never seen
          a state or (state,action) tuple
        """
        if (state, action) in self.q_values:
            return self.q_values[(state, action)]
        return 0.0

    def get_value(self, obs):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.
        """
        actions = [i for i in range(self.num_actions)]
        if len(actions) == 0:  # there are no possible actions
            return 0
        q_value_list = [self.get_q_value(obs, action) for action in actions]
        max_q_value = max(q_value_list)
        return max_q_value

    def get_policy(self, obs):
        """
          Computes the best action to take in a state.
        """
        actions = [i for i in range(self.num_actions)]
        if len(actions) == 0:  # there are no possible actions
            return None
        q_value_dict = {action: self.get_q_value(obs, action) for action in actions}
        max_action = max(q_value_dict, key=q_value_dict.get)
        return max_action

    def update_return(self, state, action):
        if (state, action) not in self.total_return:
            self.total_return[(state, action)] = 0.0
        rewards = [r for (s, a, r) in self.episode]
        r = sum(rewards[self.t:])  # compute the return R of the state-action pair as the sum of rewards
        self.total_return[(state, action)] = self.discount * self.total_return[(state, action)] + r

    def generate_episode(self):
        obs = self.env.reset()
        done = False
        for t in range(MAX_EPISODE_LEN):  # for each time step
            if done:
                break

            action = self.action_callback(obs)  # select the action according to the epsilon-greedy policy
            if action is None:
                raise Exception('Error: Agent returned None action')
            new_obs, reward, done, _ = self.env.step(action)
            self.episode.append((obs, action, reward))
            obs = new_obs
        self.env.reset()

    def episode_callback(self):
        """
        The Updates after every episode.
        """
        self.q_values = {}
        self.episode = []
        self.generate_episode()
        self.t = 0
        self.total_return = {}


class MCOnPolicyAgent(MCAgents):
    def __init__(self, **args):
        super().__init__(**args)
        self.state_action_pairs = [(s, a) for (s, a, r) in self.episode]
        self.visited = {}

    def update_visited(self, state, action):
        if (state, action) not in self.visited:
            self.visited[(state, action)] = 0
        self.visited[(state, action)] += 1

    def action_callback(self, obs):
        """
        Chooses an action and return it.
        """
        actions = [i for i in range(self.num_actions)]
        if len(actions) == 0:  # there are no possible actions
            return None
        if agent_utils.flip_coin(self.epsilon):
            return random.choice(actions)
        max_action = self.get_policy(obs)
        return max_action

    def experience_callback(self, obs, action, new_obs, reward, done):
        """
        Updating the rewards.
        """
        if self.t < len(self.episode):
            obs, action, reward = self.episode[self.t]
            if (obs, action) not in self.state_action_pairs[:self.t]:
                if action is None:
                    raise Exception('Error: Agent returned None action')
                self.update_return(obs, action)
                self.update_visited(obs, action)
                if (obs, action) not in self.q_values:
                    self.q_values[(obs, action)] = 0.0
                # Update rule: Q(S,A) <- average(returns(S_t,A_t))
                self.q_values[(obs, action)] = self.total_return[(obs, action)] / self.visited[(obs, action)]
            self.t += 1

    """ Evaluation callbacks """

    def policy_callback(self, state):
        pass


class MCOffPolicyAgent(MCAgents):
    def __init__(self, **args):
        super().__init__(**args)
        self.costs = {}
        self.weight = 1

    def update_cost(self, state, action):
        if (state, action) not in self.costs:
            self.costs[(state, action)] = 0.0
        self.costs[(state, action)] = self.costs[(state, action)] + self.weight

    def update_weight(self):
        actions = [i for i in range(self.num_actions)]
        if len(actions) == 0:  # there are no possible actions
            return self.weight
        self.weight = self.weight * (1. / len(actions))

    def action_callback(self, obs):
        """
        Chooses an action and return it.
        """
        actions = [i for i in range(self.num_actions)]
        if len(actions) == 0:  # there are no possible actions
            return None
        return random.choice(actions)

    def experience_callback(self, obs, action, new_obs, reward, done):
        """
        Updating the rewards.
        """
        if action is None:
            raise Exception('Error: Agent returned None action')
        obs, action, reward = self.episode[self.t]
        self.update_return(obs, action)
        self.update_cost(obs, action)
        if (obs, action) not in self.q_values:
            actions = [i for i in range(self.num_actions)]
            for a in actions:
                self.q_values[(obs, a)] = 0.0
        Q, G = self.q_values[(obs, action)], self.total_return[(obs, action)]
        # Update rule: Q(S,A) <- Q(S,A) + (W / C(S,A))[G - Q(S,A)]
        self.q_values[(obs, action)] = Q + ((self.weight / self.costs[(obs, action)]) * (G - Q))
        if action != self.get_policy(obs):
            return
        else:
            self.update_weight()

    def episode_callback(self):
        """
        The Updates after every episode.
        """
        pass

    """ Evaluation callbacks """

    def policy_callback(self, state):
        pass

from abc import ABC
from agent_utils import *
from environments.gridworld.gridworld_utils import TERMINAL_STATE
from learning_agent import RL_Agent
import numpy as np


class TDAgent(RL_Agent, ABC):
    def __init__(self, epsilon=0.2, discount=0.9, **args):
        super().__init__(**args)
        self.epsilon = epsilon
        self.discount = discount
        self.q_values = {}
        self.terminal_states = None
        self.episodeRewards = 0

    def get_q_value(self, state, action):
        """
          Returns Q(state,action) or 0.0 if we never seen a state or (state,action) tuple
        """
        if (state, action) in self.q_values:
            return self.q_values[(state, action)]
        return 0.0

    def get_policy(self, state):
        """
          Computes the best action to take in a state.
        """
        actions = self.get_legal_actions(state)
        if len(actions) == 0:  # there are no possible actions
            return None
        q_value_dict = {action: self.get_q_value(state, action) for action in actions}
        max_action = max(q_value_dict, key=q_value_dict.get)
        return max_action

    def action_callback(self, state):
        """
          Computes the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.
          Should use transform_fn if it exist.
        """
        actions = self.get_legal_actions(state)
        # if self.mapping_fn:
        #     return self.mapping_fn(state, actions)
        if len(actions) == 0:  # there are no possible actions
            return None
        if flip_coin(self.epsilon):
            return random.choice(actions)
        max_action = self.get_policy(state)
        return max_action

    def episode_callback(self):
        self.update_alpha()
        self.stop_episode()
        self.start_episode()

    def update_alpha(self):
        """
        Updates the exploration rate in the end of each episode.
        """
        self.alpha = MIN_EXPLORATION_RATE + (MAX_EXPLORATION_RATE - MIN_EXPLORATION_RATE) * np.exp(
            -EXPLORATION_DECAY_RATE * self.episodesSoFar)  # Exploration rate decay

    def start_episode(self):
        # self.lastState = None
        # self.lastAction = None
        self.episodeRewards = 0.0

    def stop_episode(self):
        """
          Called by environment when episode is done
        """
        if self.episodesSoFar < self.num_training:
            self.accumTrainRewards += self.episodeRewards
        else:
            self.accumTestRewards += self.episodeRewards
        self.episodesSoFar += 1
        if self.episodesSoFar >= self.num_training:
            # Take off the training wheels
            self.epsilon = 0.0  # no exploration
            self.alpha = 0.0  # no learning

    def get_legal_actions(self, state):
        if self.num_actions == 6 or self.num_actions == 4:
            return [i for i in range(self.num_actions)]
        if state == TERMINAL_STATE:
            return ()
        elif state in self.terminal_states:
            return ('exit',)
        return 'up', 'left', 'down', 'right'

    def set_terminal_states(self, terminal_states):
        self.terminal_states = terminal_states


class QLearningAgent(TDAgent):
    def __init__(self, **args):
        TDAgent.__init__(self, **args)

    def get_value(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.
        """
        actions = self.get_legal_actions(state)
        if len(actions) == 0:  # there are no possible actions
            return 0
        q_value_list = [self.get_q_value(state, action) for action in actions]
        max_q_value = max(q_value_list)
        return max_q_value

    def experience_callback(self, obs, action, new_obs, reward, done):
        """"
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          Does the Q-Value update
        """
        self.episodeRewards += reward
        if (obs, action) not in self.q_values:
            self.q_values[(obs, action)] = 0.0
        cur_val = self.q_values[(obs, action)]
        max_next_q_val = self.get_value(new_obs)
        # Update rule: Q(S,A) <- Q(S,A) + alpha * [R + gamma * max{Q(S',a)} - Q(S, A)]
        self.q_values[(obs, action)] = cur_val + self.alpha * (reward + (self.discount * max_next_q_val) - cur_val)

    """ Evaluation callbacks """

    def policy_callback(self, state):
        pass


class SarsaAgent(TDAgent):
    def __init__(self, **args):
        TDAgent.__init__(self, **args)

    def experience_callback(self, obs, action, new_obs, reward, done):
        """"
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          Does the Q-Value update
        """
        next_action = self.action_callback(new_obs)

        if (obs, action) not in self.q_values:
            self.q_values[(obs, action)] = 0.0

        if (new_obs, next_action) not in self.q_values:
            self.q_values[(new_obs, next_action)] = 0.0

        cur_val = self.q_values[(obs, action)]
        next_q_val = self.q_values[(new_obs, next_action)]
        # Update rule: Q(S,A) <- Q(S,A) + alpha * [R + gamma * Q(S',A') - Q(S, A)]
        self.q_values[(obs, action)] = cur_val + self.alpha * (reward + (self.discount * next_q_val) - cur_val)

    """ Evaluation callbacks """

    def policy_callback(self, state):
        pass

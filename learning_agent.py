from abc import ABC, abstractmethod


class RL_Agent(ABC):
    def __init__(self, num_actions, theta, alpha=0.00025, gamma=0.9, mapping_fn=None):
        """
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        num_actions - number of actions in the current environment
        """
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0
        self.num_training = 1000
        """ Parameters """
        self.theta = theta
        self.alpha = alpha
        self.gamma = gamma

        self.mapping_fn = mapping_fn
        self.num_actions = num_actions

    """ Training Callbacks """

    @abstractmethod
    def action_callback(self, state):
        """
        Chooses an action and return it.
        """
        pass

    @abstractmethod
    def experience_callback(self, obs, action, new_obs, reward, done):
        """
        Updating the rewards.
        """
        pass

    @abstractmethod
    def episode_callback(self):
        """
        The Updates after every episode.
        """
        pass

    """ Evaluation callbacks """

    @abstractmethod
    def policy_callback(self, state):
        pass

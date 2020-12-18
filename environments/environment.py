from abc import ABC, abstractmethod


class AbstractEnvironment(ABC):
    @abstractmethod
    def reset(self):
        """
          Resets the current state to the start state
        """
        pass

    @abstractmethod
    def render(self):
        """
        Renders the environment.
        """
        pass

    @abstractmethod
    def step(self, action):
        """
          Performs the given action in the current
          environment state and updates the environment.

          Returns (new_obs, reward, done, info)
        """
        pass


class Environment(AbstractEnvironment):
    @abstractmethod
    def get_current_state(self):
        """
        Returns the current state of environment
        """
        pass

    @abstractmethod
    def get_possible_actions(self, state):
        """
          Returns possible actions the agent
          can take in the given state. Can
          return the empty list if we are in
          a terminal state.
        """
        pass

    def is_terminal(self):
        """
          Has the environment entered a terminal
          state? This means there are no successors
        """
        state = self.get_current_state()
        actions = self.get_possible_actions(state)
        return len(actions) == 0

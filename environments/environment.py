from abc import ABC, abstractmethod


class Environment:
    @abstractmethod
    def get_current_state(self):
        """
        Returns the current state of enviornment
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

    @abstractmethod
    def step(self, action):
        """
          Performs the given action in the current
          environment state and updates the enviornment.

          Returns a (reward, nextState) pair
        """
        pass

    @abstractmethod
    def reset(self):
        """
          Resets the current state to the start state
        """
        pass

    @abstractmethod
    def is_terminal(self):
        """
          Has the environment entered a terminal
          state? This means there are no successors
        """
        state = self.get_current_state()
        actions = self.get_possible_actions(state)
        return len(actions) == 0

__author__ = 'sarah'

from Search.problem import Problem

class GymProblem(Problem):

    """Problem superclass
       supporting COMPLETE
    """
    def __init__(self, env, constraints=[]):
        super().__init__(env.s, constraints)
        self.env = env

    # get the actions that can be applied at the current node
    def get_applicable_actions(self, node):
        action_list = self.env.P[node.state.__repr__()].keys()
        return action_list

    # get (all) succesor states of an action and their
    def get_successors(self, action, state):

        action_list = self.env.P[state.__repr__()]
        [prob, next_state, reward, done] = self.env.P[state][action][0]
        self.done = done
        return next_state

    def get_action_cost(self, action, state):
        return 1

    def is_goal_state(self, state):
        if 0:
            return True
        else:
            return False





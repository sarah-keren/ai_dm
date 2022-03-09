__author__ = 'sarah'

from Search.problem import Problem

class GymProblem(Problem):

    """Problem superclass
       supporting COMPLETE
    """
    def __init__(self, env, constraints=[]):
        super().__init__(env.s, constraints)
        self.env = env
        self.done = False


    # value of a node
    def evaluate(self, node, use_cost_as_value=True):
        if use_cost_as_value:
            return node.get_path_cost(self)
        # use value
        else:
            return node.get_path_value(self)

    # return whether val_a is better or equal to val_b in the domain
    def is_better_or_equal(self, val_a, val_b):
        if val_a > val_b or val_a == val_b:
            return True
        else:
            return False

    # get the actions that can be applied at the current node
    def get_applicable_actions(self, node):
        actions_trans = self.env.P[node.state]
        return actions_trans

    # apply the action and return the next state
    def get_successor_state(self, action, state):
        [prob, next_state, reward, done] = self.env.P[state][action][0]
        self.done = done
        return next_state

    def get_action_cost(self, action, state):
        return 1

    def is_goal_state(self, state):
        if self.done:
            return True
        else:
            return False





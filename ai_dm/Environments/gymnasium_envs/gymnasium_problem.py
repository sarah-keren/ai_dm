__author__ = 'sarah'

from ai_dm.Base.problem import Problem
import ai_dm.Search.utils as utils

# Wrapping Gymnasium problems for which the probability function P can be accessed.
# Examples include taxi, frozen_lake and Cliff Walking
class GymnasiumProblemP(Problem):

    """Problem superclass
       supporting Gymnasium problems
    """
    def __init__(self, env, init_state, constraints=[], action_cost=1):
        super().__init__(init_state, constraints)
        self.env = env
        self.counter = 0
        self.action_cost = action_cost
        self.action_costs = None

    # get the actions that can be applied at the current node
    def get_applicable_actions_at_node(self, node):
        action_list = self.env.P[node.state.get_key()].keys()
        return action_list

    # get the actions that can be applied at the current node
    def get_applicable_actions_at_state(self, state):
        action_list = self.env.P[state.get_key()].keys()
        return action_list

    # get (all) succesor states of an action and their
    def get_successors(self, action, node):

        #action_list = self.env.P[node.state.__repr__()]
        successor_nodes = []
        transitions = self.env.P[node.state.__str__()][action]
        action_cost = self.get_action_cost(action, node.state)
        for prob, next_state_key, reward, done in transitions:
            info={}
            info['prob'] = prob
            info['reward'] = reward
            next_state = utils.State(next_state_key, done)
            successor_node = utils.Node (state=next_state, parent=node, action=action, path_cost=node.path_cost + action_cost, info=info)
            successor_nodes.append(successor_node)

        return successor_nodes

    def set_action_costs(self, action_costs):
        self.action_cost = action_costs

    def get_action_cost(self, action, state):
        if self.sction_costs:
            return self.action_costs[action]
        else:
            return self.action_cost

    def is_goal_state(self, state):
        if state.is_terminal:
            return True
        else:
            return False

    def apply_action(self, action):
        return self.env.step(int(action))

    # reset environment and return initial state
    def reset_env(self):
        return self.env.reset()[0]




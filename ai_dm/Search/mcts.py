import ai_dm.Search.utils as utils
import ai_dm.Search.defs as defs
import ai_dm.Search.heuristic as heuristic
import ai_dm.Search.constraint as constraint
import logging, time


# based on code from https://ai-boson.github.io/mcts/
import numpy as np
from collections import defaultdict

class MCTSNode(utils.Node):

    def __init__(self, state, parent=None, action=None, path_cost=0, info=None, parent_action=None):
        super().__init__(state, parent, action, path_cost, info)
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions()
        return

    def untried_actions(self):
        return None
        #return self.problem.get_applicable_actions(self)

    def n(self):
        return self._number_of_visits

    def is_terminal_node(self):
        return self.state.is_game_over()

def mcts(problem, comp_resources, selection_policy, expansion_policy, rollout_policy):
    root_node = MCTSNode(problem.get_current_state(), None, None, 0, None, None)
    print(root_node)
    while not are_resources_exhausted(comp_resources):
        leaf = select(root_node)
        child = expand(leaf)
        simulation_result = simulate(child) #rollout
        backpropagate(simulation_result, child)
    return best_action(root_node)
def are_resources_exhausted(comp_resources):
    return False

def expand(node):
    #action = self._untried_actions.pop()
    #next_state = self.state.move(action)
    #child_node = MonteCarloTreeSearchNode(
    #    next_state, parent=self, parent_action=action)
    #self.children.append(child_node)
    #return child_node
    return node

def select(node, c_param=0.1):
    #choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
    #return self.children[np.argmax(choices_weights)]
    return node

def simulate(node):
    return 0
    #current_rollout_state = node.state

    #while not current_rollout_state.is_game_over():
    #    possible_moves = current_rollout_state.get_legal_actions()

    #    action = rollout_policy(possible_moves)
    #    current_rollout_state = current_rollout_state.move(action)
    #return current_rollout_state.game_result()

def backpropagate(node, result):
    #self._number_of_visits += 1.
    #self._results[result] += 1.
    #if self.parent:
    #    self.parent.backpropagate(result)
    return node

def rollout_policy(possible_moves):
    return possible_moves[np.random.randint(len(possible_moves))]

def selection_policy(node):
    current_node = node
    while not current_node.is_terminal_node():

        if not current_node.is_fully_expanded():
            return current_node.expand()
        else:
            current_node = current_node.best_child()
    return current_node

def expansion_policy(node):
    current_node = node
    while not current_node.is_terminal_node():

        if not current_node.is_fully_expanded():
            return current_node.expand()
        else:
            current_node = current_node.best_child()
    return current_node

def best_action(node):
    #simulation_no = 100

    #for i in range(simulation_no):
    #    v = self._tree_policy()
    #    reward = v.rollout()
    #    v.backpropagate(reward)
    #return self.best_child(c_param=0.)
    return node.action()
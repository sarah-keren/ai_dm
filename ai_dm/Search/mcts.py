import ai_dm.Search.utils as utils
import ai_dm.Search.defs as defs
import ai_dm.Search.heuristic as heuristic
import ai_dm.Search.constraint as constraint
import logging, time


# based on code from https://ai-boson.github.io/mcts/
import numpy as np
from collections import defaultdict

class MCTSNode(utils.Node):

    def __init__(self, state, applicable_actions, parent=None, action=None, path_cost=0, info=None, parent_action=None):
        super().__init__(state, parent, action, path_cost, info)
        self.parent_action = parent_action
        # todo: support settings with continous spaces
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        #self._results[1] = 0
        #self._results[-1] = 0
        self._untried_actions = applicable_actions
        return

    def n(self):
        return self._number_of_visits

    def is_leaf(self):
        return True




def mcts(problem, comp_resources, selection_policy, expansion_policy, rollout_policy):

    # initialize the search tree
    root_node = MCTSNode(problem.get_current_state(), problem.get_applicable_actions_at_state(problem.get_current_state()),  None, None, 0, None, None)

    # perform the search
    while not are_resources_exhausted(comp_resources):
        comp_resources.update()
        # use selection (tree) policy to choose the next leaf node to expand
        leaf = select(root_node, selection_policy)
        # choose which child of the selected leaf to expand (i.e. perform a simulation from)
        expanded_child = expand(leaf, expansion_policy)
        # perform a single simulation according to the rollout_policy from the expanded child node
        simulation_result = simulate(expanded_child, problem, rollout_policy)
        # update the tree with the current values
        backpropagate(simulation_result, expanded_child)

    return best_action(root_node)
def are_resources_exhausted(comp_resources):
    return False


def uct_selection_policy(node, params):
    c_param=0.1
    value = node.state.value()
    # compute uct values of all the node's children
    children_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(node.n()) / c.n())) for c in node.children]
    # return the best nodes
    return node.children[np.argmax(children_weights)]


# recursively traverse the tree until a leaf node (of the current MCTS tree) is reached.
def select(mcts_node, selection_policy):
    if mcts_node.is_leaf():
        return mcts_node
    else:
        return select(selection_policy(mcts_node))

def expand(node, policy):
    #action = self._untried_actions.pop()
    #next_state = self.state.move(action)
    #child_node = MonteCarloTreeSearchNode(
    #    next_state, parent=self, parent_action=action)
    #self.children.append(child_node)
    #return child_node
    return node

def simulate(init_node, problem, rollout_policy):
    cur_node = init_node
    while not problem.is_terminal(cur_node.state):
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

def default_rollout_policy(possible_moves):
    return possible_moves[np.random.randint(len(possible_moves))]

def default_selection_policy(node):
    current_node = node
    while not current_node.is_terminal_node():

        if not current_node.is_fully_expanded():
            return current_node.expand()
        else:
            current_node = current_node.best_child()
    return current_node

def default_expansion_policy(node):
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
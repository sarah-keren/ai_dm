__author__ = 'sarah'

from abc import ABC, abstractmethod
import Search.utils as utils


class Problem (ABC):

    """Problem superclass
       supporting COMPLETE
    """

    def __init__(self, initial_state, constraints):
        self.initial_state = initial_state
        self.constraints = constraints

    # returns the value of a node
    @abstractmethod
    def evaluate(self, node):
        pass

    # return whether val_a is better or equal to val_b in the domain
    @abstractmethod
    def is_better_or_equal(self, val_a, val_b):
        pass

    # return the actions that are applicable in the current state
    @abstractmethod
    def get_applicable_actions(self, node):
        pass

    # return the successor state that will result from applying the action (without changing the state)
    @abstractmethod
    def get_successor_state(self, action, state):
        pass

    # return the action's cost
    @abstractmethod
    def get_action_cost(self, action, state):
        pass


    # does the state represent a goal state
    @abstractmethod
    def is_goal_state(self, state):
        pass


    # is the state valid in the domain
    def is_valid(self, state):

        # if there are no constraints - return True
        if self.constraints is None:
            print('No constraints')
            return True
        # check all constraints - if one is violated, return False
        for constraint in self.constraints:
            if not constraint.is_valid(state):
                return False
        # non of the constraints have been violated
        return True

    # get all successors for cur_node
    def successors(self, cur_node, cleanup=False):

        # the state represented by the node
        cur_state = cur_node.state

        # get the actions that can be applied to this node
        action_list = self.get_applicable_actions(cur_node)
        if action_list is None:
            return None

        # remove the modifications that violate the constraints
        successor_nodes = []
        for action in action_list:

            successor_state = self.get_successor_state(action, cur_state)
            action_cost = self.get_action_cost(action, cur_state)
            successor_node = utils.Node(successor_state, cur_node, action, cur_node.path_cost+action_cost, cur_node.state)

            valid = True
            # iterate through the constraints to see if the current action or successor states violate them
            for constraint in self.constraints:
                if not constraint.is_valid(successor_node, action):
                    valid = False
                    break

            # apply the action
            if valid:
                ''' add the succesor node specifying:
                    successor_state
                    cur_node (ancestor)
                    action (the applied action)
                    node.path_cost+action.cost (the accumulated cost)
                '''
                successor_nodes.append(successor_node)

        if cleanup:
            cur_state.clean_up()

        return successor_nodes


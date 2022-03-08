__author__ = 'sarah'

from abc import ABC, abstractmethod

class Problem (ABC):

    """Problem superclass
       supporting COMPLETE
    """

    def __init__(self, initial_state, constraints ):
        self.initial_state = initial_state
        self.constraints = constraints

    # returns the value of a state
    @abstractmethod
    def evaluate(self, state):
        pass

    # return whether val_a is better then val_b in the domain
    def is_better(self, val_a, val_b):
        raise NotImplementedError

    # is the state valud in the domain
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
    def successors(self, cur_node, cleanup = True):


        # the state represented by the node
        cur_state = cur_node.state

        # get the actions that can be applied to this node
        action_list = self.get_applicable_actions(cur_node)
        if action_list is None:
            return None

        # remove the modifications that violate the constraints
        successor_nodes = []
        for action in action_list:

            successor_state = action.apply(cur_state)
            successor_node = search.Node(successor_state, cur_node, action, cur_node.path_cost+action.cost, cur_node.state)

            valid = True
            # iterate through the constraints to see if the current action or successor states violate them
            for constraint in self.constraints:
                if not constraint.is_valid(successor_node, action):
                    valid = False
                    break

            # apply the modifications
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


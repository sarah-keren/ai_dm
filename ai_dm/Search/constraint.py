__author__ = 'sarah'

from abc import ABC, abstractmethod
import ai_dm.Search.utils as utils


class Constraint (ABC):

    # returns whether it is valid to apply the action at the state
    # represented by the node (used by the successor function)
    @abstractmethod
    def is_valid(self, node, action=None):
        pass


class DepthConstraint(Constraint):

    '''
    the number of allowed actions
    '''
    def __init__(self, depth):
        self.depth = depth

    def __repr__(self):
        return "%d"%self.depth

    '''
    check the budget constraints have not been violated
    '''
    def is_valid(self, node, action=None):

        if len(node.get_transition_path()) <= self.depth:
            return True
        return False




class ComputationalResources(ABC):

    """Problem superclass
       supporting COMPLETE
    """
    # return whether there are any remaining resources
    @abstractmethod
    def are_exhausted(self):
        pass


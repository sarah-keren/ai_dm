import time


class ComputationalResources:

    def __init__(self, iteration_bound=None, time_bound=None):

        # creating the initial state object
        self.iteration_bound = iteration_bound
        if self.iteration_bound:
            self.iteration_count = 0

        self.time_bound = time_bound
        if self.time_bound:
            self.init_time = time.time()
            self.current_time = self.init_time

            # return whether there are any remaining resources
    def are_exhausted(self, cur_iteration= None, cur_time = None):

        if self.iteration_bound:
            if self.iteration_count > cur_iteration:
                return True

        if self.time_bound:
            if self.current_time - self.init_time > self.time_bound:
                return True

        return False

    def update(self):

        if self.iteration_bound:
            self.iteration_count += 1

        if self.time_bound:
            self.current_time = time.time()
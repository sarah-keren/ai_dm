import gym
import numpy as np
from queue import Queue

# Expert policy for Taxi-v2 domain

# Taxi Actions:
# There are 6 discrete deterministic actions:
# - 0: move south
# - 1: move north
# - 2: move east 
# - 3: move west 
# - 4: pickup passenger
# - 5: dropoff passenger

class Taxi_Expert:
    def __init__(self, env):
        self.env = env
        self.desc = env.desc # array holding map layout
        self.locs = env.locs # locations of passenger pickup or dropoff X's on map
        self.num_rows = len(self.desc) - 2
        self.num_cols = len(self.desc[0][1:-1:2])
        self.max_row = self.num_rows - 1
        self.max_col = self.num_cols - 1
        self.shortest_path_trees = []
        for loc in self.locs:
            self.shortest_path_trees.append(self.shortest_path_tree(loc))

    # produce a shortest path tree for the shortest ways to get to the location specified    
    def shortest_path_tree(self, loc):
        spt = np.empty((self.num_rows,self.num_cols))
        spt[:] = np.nan
        spt[loc] = -1 # once taxi arrives at destination, there is no better action to take
        q = Queue()
        q.put(loc)
        while not q.empty(): # until no entries in the array are still nan
            (row, col) = q.get() # current location we are exploring the neighbors of
            for action in range(4): 
                if action == 0:
                    new_loc = (min(row + 1, self.max_row), col)
                    if np.isnan(spt[new_loc]):
                        spt[new_loc] = 1
                        q.put(new_loc)
                elif action == 1:
                    new_loc = (max(row - 1, 0), col)
                    if np.isnan(spt[new_loc]):
                        spt[new_loc] = 0
                        q.put(new_loc)
                elif action == 2 and self.desc[1 + row, 2 * col + 2] == b":":
                    new_loc = (row, min(col + 1, self.max_col))
                    if np.isnan(spt[new_loc]):
                        spt[new_loc] = 3
                        q.put(new_loc)
                elif action == 3 and self.desc[1 + row, 2 * col] == b":":
                    new_loc = (row, max(col - 1, 0))
                    if np.isnan(spt[new_loc]):
                        spt[new_loc] = 2
                        q.put(new_loc)
        spt = np.rint(spt)
        return spt

    # get expert action, given a state
    def get_action(self, state):
        taxi_row, taxi_col, pass_loc, dest_idx = self.env.decode(int(state))
        taxi_loc = (taxi_row, taxi_col)
        if pass_loc == len(self.locs) and taxi_loc == self.locs[dest_idx]:
            return 5 # dropoff
        elif pass_loc == len(self.locs) and taxi_loc != self.locs[dest_idx]:
            return int(self.shortest_path_trees[dest_idx][taxi_loc]) # move towards destination
        elif pass_loc < len(self.locs) and taxi_loc == self.locs[pass_loc]:
            return 4 # pickup
        elif pass_loc < len(self.locs) and taxi_loc != self.locs[pass_loc]:
            return int(self.shortest_path_trees[pass_loc][taxi_loc]) # move towards passenger
        else:
            print("Not a valid state.")


# Processor class for processing Taxi state data
# Currently only works with maps that have 4 dropoff locations

class Taxi_Processor:
    def __init__(self, env):
        from sklearn.preprocessing import MinMaxScaler
        desc = env.desc
        self.env = env
        self.num_rows = len(desc) - 2
        self.num_cols = len(desc[0][1:-1:2])
        self.locs = env.locs # locations of passenger pickup or dropoff X's on map
        min_data = [0 for _ in range(2 + 2 * len(self.locs) + 1)]
        max_data = [self.num_rows - 1, self.num_cols - 1] + [1 for _ in range(2 * len(self.locs) + 1)]
        self.scaler = MinMaxScaler().fit([min_data, max_data])


    def taxi_expert_to_model(self, state):
        # expert_obs = state
        expert_obs = list(self.env.decode(state))
        processed = [0 for _ in range(2 + 2 * len(self.locs) + 1)] # (2 for taxi row and col, 2 * num locations for one-hot encodings of taxi and passenger locations, one more for if the passenger is in the taxi)
        processed[0] = expert_obs[0]
        processed[1] = expert_obs[1]
        processed[2 + expert_obs[2]] = 1
        processed[7 + expert_obs[3]] = 1
        # return processed
        return self.scaler.transform([processed])[0]

    def taxi_model_to_expert(self, model_obs):
        inverted = self.scaler.inverse_transform([model_obs])[0]
        unprocessed = [0 for _ in range(4)]
        unprocessed[0] = inverted[0]
        unprocessed[1] = inverted[1]
        if inverted[2] == 1:
            unprocessed[2] = 0
        elif inverted[3] == 1:
            unprocessed[2] = 1
        elif inverted[4] == 1:
            unprocessed[2] = 2
        elif inverted[5] == 1:
            unprocessed[2] = 3
        elif inverted[6] == 1:
            unprocessed[2] = 4
        if inverted[7] == 1:
            unprocessed[3] = 0
        elif inverted[8] == 1:
            unprocessed[3] = 1
        elif inverted[9] == 1:
            unprocessed[3] = 2
        elif inverted[10] == 1:
            unprocessed[3] = 3
        # return unprocessed
        return self.env.encode(*unprocessed)
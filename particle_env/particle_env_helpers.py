"""
Set of discretization functions for the particle env. Discrete_listener_state (not enough info) and discretize_bins (too large) aren't very good
"""
import numpy as np

def discretize_speaker_state(observation):
    """ based on the index of 0.65 in the landmark """  
    idx = np.where(observation == 0.65)[0][0]
    return np.array([idx]) 

def discretize_listener_state(listener_info):
    """ TODO: do something more useful
    Turn a continuous state space into something that can be put into a table
    For now use relative quadrant with respect to landmark at the center, e.g.:
    (+,+) -> 0, (-,+) -> 1, (-,-) -> 2, (+, -) -> 3
    Then do this for each landmark
        |
    ----|----
        |
    """
    point_int_map = {(1,1): 0, (-1,1): 1, (-1,-1): 2, (1, -1): 3}

    def get_quadrant(rel_x, rel_y):
        sign_x, sign_y = np.sign(rel_x), np.sign(rel_y)
        return point_int_map[(sign_x, sign_y)]

    # relation to landmark (a)
    state_a = get_quadrant(listener_info[2], listener_info[3])
    # relation to landmark (b) 
    state_b = get_quadrant(listener_info[4], listener_info[5])
    # # relation to landmark (c) 
    state_c = get_quadrant(listener_info[6], listener_info[7])

    speaker_comm = listener_info[8:]
    comm = np.argmax(speaker_comm)

    return np.array([state_a, state_b, state_c, comm])

def discrete_bins(listener_info):
    num_bins = 10

    # assume values go from 0 to 4, 20 bins
    def bin(x):
        return int(round(round(x, 1) * 5) + 10)

    bins = np.array([bin(listener_info[i]) for i, _ in enumerate(listener_info) if i >= 2 and i <= 7])
    speaker_comm = listener_info[8:]

    comm = np.argmax(speaker_comm)

    return np.append(bins, comm)

# agent2 = QLearningAgent(num_actions=5, dim=(5,3,3, 3), mapping_fn=peh.discrete_gridworld) # listener

def discrete_gridworld(listener_info):
    # landmark positions start @ [-1,1]. They are RELATIVE to the agent
    num_rows = 3
    num_cols = 3
    box_size = 4 / num_rows

    def get_grid_location(landmark):
        (x, y) = landmark             
        # map range [-2,2] to [0,4]
        x = x + 2
        y = y + 2

        x, y = x // box_size, y // box_size
        # handle out of bounds
        if x >= num_rows: x = num_rows-1
        if x < 0: x = 0
        if y >= num_rows: y = num_rows-1 
        if y < 0: y = 0 

        return (int(x), int(y))

    land_a = get_grid_location(listener_info[2:4])
    land_b = get_grid_location(listener_info[4:6])
    land_c = get_grid_location(listener_info[6:8])

    discrete_state = land_a + land_b + land_c # this is num_rows^6 landmark states

    # if multiagent
    # speaker_comm = listener_info[8:]
    # comm = np.argmax(speaker_comm)

    # return np.append(discrete_state, comm) # this is num_rows^6 * 3 landmark states

    # if single agent
    return discrete_state





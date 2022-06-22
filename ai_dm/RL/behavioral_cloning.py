import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from queue import Queue


""" 
Behavioral Cloning Agent 

Args:
env: agent's environment
get_expert_action: given a state, get the action the expert would choose for it
obs_per_iteration: observations to collect from training each iteration
convert_expert_to_model: function that processes environment state into a form the model can learn from
convert_model_to_expert: inverse of convert_expert_to_model
model: the learning model to use; default is a scikit Gradient Boosting model
iterations: number of iterations to run: 1 iteration is basic behavioral cloning, and >1 iteration is DAgger
max_episode_length: when collecting model data, terminate an agent episode if it exceeds this length

Implements:
collect_expert_data: run the expert policy and return its states and actions
get_model_data: run the model policy and return its states, actions, and rewards (rewards for each completed episode)
expert_relabel: relabel a 2D list of model states and return the actions the expert would have taken
train_dagger: iteratively collect data and train from it
predict: given state(s), use model to predict what the action should be

"""

class BehavioralCloningAgent(object):
    def __init__(self, env, get_expert_action, obs_per_iteration,
            convert_expert_to_model = lambda x: x, # these functions should be observation to observation
            convert_model_to_expert = lambda x: x,
            model = GradientBoostingClassifier(n_estimators=300, learning_rate=.01, random_state=0),
            iterations = 1,
            max_episode_length = 50):
        self.env = env
        self.get_expert_action = get_expert_action
        self.obs_per_iteration = obs_per_iteration
        self.model = model
        self.expert_to_model = convert_expert_to_model
        self.model_to_expert = convert_model_to_expert
        self.iterations = iterations
        self.max_episode_length = max_episode_length

    # collect a list of states the expert visited and list of actions the expert made in those states
    def collect_expert_data(self, num_observations = None):
        if num_observations is None:
            num_observations = self.obs_per_iteration
        data = []
        labels = []
        while len(data) < num_observations:
            self.env.reset()
            episode_done = False
            while not episode_done and len(data) < num_observations:
                # data.append(list(self.env.decode(self.env.s)))
                data.append(self.env.s)
                ac = self.get_expert_action(self.env.s)
                labels.append(ac)
                _, _, d, _ = self.env.step(ac)
                episode_done = d
        return data, labels

    def get_model_data(self, num_observations = None):
        if num_observations is None:
            num_observations = self.obs_per_iteration
        data = []
        labels = []
        episode_rewards = []
        episode_length = 0
        while len(data) < num_observations:
            self.env.reset()
            episode_done = False
            episode_length = 0
            episode_reward = 0
            while not episode_done and len(data) < num_observations and episode_length <= self.max_episode_length:
                # state_data = self.expert_to_model(list(self.env.decode(self.env.s)))
                state_data = self.expert_to_model(self.env.s)
                data.append(list(state_data))
                ac = self.model.predict([state_data])[0]
                labels.append(ac)
                _, r, d, _ = self.env.step(ac)
                episode_done = d
                episode_length += 1
                episode_reward += r
            episode_rewards.append(episode_reward)
        return (data, labels, episode_rewards)

    def expert_relabel(self, data):
        expert_labels = []
        for d in data:
            # state = self.env.encode(*self.model_to_expert(d))
            state = self.model_to_expert(d)
            expert_labels.append(self.get_expert_action(int(state)))
        return expert_labels


    def train_dagger(self):
        train_data, train_labels = self.collect_expert_data()
        train_data = [self.expert_to_model(d) for d in train_data]
        for i in range(self.iterations - 1):
            print("Running iteration:", str(i))
            self.model.fit(train_data, train_labels)
            model_data, _, episode_rewards = self.get_model_data()
            print("Average reward per episode:", str(sum(episode_rewards) / len(episode_rewards)))
            expert_labels = self.expert_relabel(model_data)
            train_data = np.concatenate((train_data, np.array(model_data)))
            train_labels = train_labels + expert_labels
        self.model.fit(train_data, train_labels)

    # data should be a 2D list or array
    def predict(self, data, in_state_form = True):
        if in_state_form:
            return self.model.predict([self.expert_to_model(d) for d in data])
        else:
            return self.model.predict(data)


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


def test_taxi():

    import gym
    from ai_dm.Environments.gym_envs.taxi_utils import Taxi_Expert
    from ai_dm.Environments.gym_envs.taxi_utils import Taxi_Processor

    env = gym.make("Taxi-v3").env
    env.reset()
    expert = Taxi_Expert(env)
    processor = Taxi_Processor(env)

    bc_agent = BehavioralCloningAgent(env, expert.get_action, 2000, processor.taxi_expert_to_model, processor.taxi_model_to_expert, iterations = 10)
    bc_agent.train_dagger()

if __name__ == "__main__":
    test_taxi()

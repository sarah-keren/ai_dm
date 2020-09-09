import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingClassifier

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


def test_taxi():

    import gym
    from taxi_utils import Taxi_Expert
    from taxi_utils import Taxi_Processor

    env = gym.make("Taxi-v2").env
    env.reset()
    expert = Taxi_Expert(env)
    processor = Taxi_Processor(env)

    bc_agent = BehavioralCloningAgent(env, expert.get_action, 2000, processor.taxi_expert_to_model, processor.taxi_model_to_expert, iterations = 10)
    bc_agent.train_dagger()

if __name__ == "__main__":
    test_taxi()
        
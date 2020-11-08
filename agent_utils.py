import random

# ==================== agents ==================== #
Q_LEARNING_AGENT = "q"
SARSA_AGENT = "sarsa"
MONTE_CARLO_ON_POLICY = "mc_on_policy"
MONTE_CARLO_OFF_POLICY = "mc_off_policy"
# ====================== env ===================== #
TAXI = "taxi"
GRID_WORLD = "grid_world"
TAXI_MODEL_IRRELEVANCE_ABSTRACTION = "taxi_model_irrelevance_abstraction"
# ====================== params ===================== #
GAMMA = 0.9  # ............. -d : discount factor, default=0.9
LIVING_REWARD = -1  # ...... -r : Reward for living for a time step, default=0.0
NOISE = 0.3  # ............. -n : How often action results in unintended direction, default=0.2
EPSILON = 0.2  # ........... -e : Chance of taking a random action in q-learning, default=0.3
ALPHA = 0.2  # ............. -l : TD learning rate, default=0.5
ITERATIONS = 10  # ......... -i : Number of rounds of value iteration, default=10
EPISODES = 100  # ............-k : Number of episodes of the MDP to run, default=1
AGENT = Q_LEARNING_AGENT  # . -a : Agent type - see above, default=random
DISPLAY = True

MAX_EXPLORATION_RATE = 1
MIN_EXPLORATION_RATE = 0.01
EXPLORATION_DECAY_RATE = 0.001


# ================================================ #

def flip_coin(p):
    r = random.random()
    return r < p

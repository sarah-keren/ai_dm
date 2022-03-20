from MultiTaxiLib.taxi_environment import TaxiEnv
from AI_agents.Environments.gym_problem import GymProblem
from AI_agents.Search.best_first_search import best_first_search, breadth_first_search, depth_first_search, a_star
import AI_agents.Search.utils as utils
import AI_agents.Search.defs as defs
import AI_agents.Search.heuristic as heuristic



def main_multi_taxi():

    # define the environment
    multi_taxi_env = TaxiEnv()
    multi_taxi_env.reset()
    taxi_env.render()



if __name__ == "__main__":
    main_multi_taxi()

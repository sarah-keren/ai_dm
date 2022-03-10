import gym
from Environments.gym_problem import GymProblem
from Search.best_first_search import best_first_search
import Search.utils as utils
import Search.defs as defs


def main():

    # define the environment
    taxi_env = gym.make("Taxi-v3").env
    taxi_env.reset()
    taxi_env.render()

    # create a wrapper of the environment to the search
    taxi_p = GymProblem(taxi_env)

    # perform BFS
    optimal_path = best_first_search(problem=taxi_p, frontier=utils.FIFOQueue(), closed_list=utils.ClosedListOfSequences(), termination_criteria=utils.TerminationCriteriaGoalStateReached(), prune_func=None, log=True,
                          log_file=None, iter_limit=defs.NA, time_limit=defs.NA, use_search_node_for_evaluation=False)
    print(optimal_path)



if __name__ == "__main__":
    main()
import gym
from AI_agents.Environments.gym_problem import GymProblem
from AI_agents.Search.best_first_search import best_first_search, breadth_first_search, depth_first_search
import AI_agents.Search.utils as utils
import AI_agents.Search.defs as defs


def main_taxi_bfs():

    # define the environment
    taxi_env = gym.make("Taxi-v3").env
    taxi_env.reset()
    #init_state = taxi_env.encode(0, 4, 4, 1) # (taxi row, taxi column, passenger index, destination index)
    init_state = taxi_env.encode(0, 3, 4, 1)  # (taxi row, taxi column, passenger index, destination index)
    taxi_row, taxi_col, pass_idx, dest_idx = taxi_env.decode(init_state)
    print(taxi_row)
    taxi_env.unwrapped.s = init_state
    print("State:", init_state)
    taxi_env.render()

    # dropping off the passenger
    #observation, reward, done, info = taxi_env.step(5)
    #print(done)
    #taxi_env.render()


    # create a wrapper of the environment to the search
    taxi_p = GymProblem(taxi_env, taxi_env.unwrapped.s)

    # perform BFS
    [best_value, best_node, best_plan, explored_count, ex_terminated, results_log] = best_first_search(problem=taxi_p,
                                                                                                       frontier=utils.FIFOQueue(),
                                                                                                       closed_list=utils.ClosedListOfSequences(),
                                                                                                       termination_criteria=utils.TerminationCriteriaGoalStateReached(),
                                                                                                       evaluation_criteria=utils.EvaluationCriteriaGoalCondition(),
                                                                                                       prune_func=None,
                                                                                                       log=True, log_file=None,
                                                                                                       iter_limit=defs.NA,
                                                                                                       time_limit=defs.NA,
                                                                                                       )
    print(best_plan)
    for action_id in best_plan:
        taxi_p.apply_action(action_id)
        taxi_p.env.render()


def main_taxi_dfs_exp():


    # define the environment
    taxi_env = gym.make("Taxi-v3").env
    taxi_env.reset()
    #init_state = taxi_env.encode(0, 4, 4, 1) # (taxi row, taxi column, passenger index, destination index)
    init_state = taxi_env.encode(3, 3, 4, 3)  # (taxi row, taxi column, passenger index, destination index)
    taxi_row, taxi_col, pass_idx, dest_idx = taxi_env.decode(init_state)
    taxi_env.unwrapped.s = init_state
    print("State:", init_state)
    taxi_env.render()

    # dropping off the passenger
    #observation, reward, done, info = taxi_env.step(5)
    #print(done)
    #taxi_env.render()


    # create a wrapper of the environment to the search
    taxi_p = GymProblem(taxi_env, taxi_env.unwrapped.s)


    # perform BFS
    [best_value, best_node, best_plan, explored_count, ex_terminated, results_log] = depth_first_search(problem=taxi_p,
                                                                                                        log=True,
                                                                                                        log_file=None,
                                                                                                        iter_limit=defs.NA,
                                                                                                        time_limit=defs.NA,
                                                                                                        )


    print(best_plan)
    for action_id in best_plan:
        taxi_p.apply_action(action_id)
        taxi_p.env.render()

def main_taxi_bfs_exp():

    # define the environment
    taxi_env = gym.make("Taxi-v3").env
    taxi_env.reset()
    #init_state = taxi_env.encode(0, 4, 4, 1) # (taxi row, taxi column, passenger index, destination index)
    init_state = taxi_env.encode(0, 3, 4, 1)  # (taxi row, taxi column, passenger index, destination index)
    taxi_row, taxi_col, pass_idx, dest_idx = taxi_env.decode(init_state)
    taxi_env.unwrapped.s = init_state
    print("State:", init_state)
    taxi_env.render()

    # dropping off the passenger
    #observation, reward, done, info = taxi_env.step(5)
    #print(done)
    #taxi_env.render()


    # create a wrapper of the environment to the search
    taxi_p = GymProblem(taxi_env, taxi_env.unwrapped.s)


    # perform BFS
    [best_value, best_node, best_plan, explored_count, ex_terminated, results_log] = breadth_first_search(problem=taxi_p,
                                                                                                          log=True,
                                                                                                          log_file=None,
                                                                                                          iter_limit=defs.NA,
                                                                                                          time_limit=defs.NA,
                                                                                                          )


    print(best_plan)
    for action_id in best_plan:
        taxi_p.apply_action(action_id)
        taxi_p.env.render()


def main_test():

    # define the environment
    copy_env = gym.make('CartPole-v1')
    copy_env.reset()
    copy_env.render()

    # create a wrapper of the environment to the search
    copy_p = GymProblem(copy_env)

    # perform BFS
    optimal_path = best_first_search(problem=copy_p, frontier=utils.FIFOQueue(), closed_list=utils.ClosedListOfSequences(), termination_criteria=utils.TerminationCriteriaGoalStateReached(), prune_func=None, log=True,
                                     log_file=None, iter_limit=defs.NA, time_limit=defs.NA, use_search_node_for_evaluation=False)
    print(optimal_path)






if __name__ == "__main__":
    # main_test()
    #main_taxi_bfs_exp()
    main_taxi_dfs_exp()
    #main_taxi_bfs()

    #main_taxi_dfs()

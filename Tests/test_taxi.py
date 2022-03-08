import gym
from Environments.gym_problem import GymProblemWrapper
from Search.best_first_search import best_first_search
def main():

    # define the environment
    taxi_env = gym.make("Taxi-v3").env
    taxi_env.reset()
    taxi_env.render()

    # create a wrapper of the environment to the search
    taxi_p = GymProblemWrapper(taxi_env)

    # perform the search
    optimal_path = best_first_search(taxi_p)
    print(optimal_path)



if __name__ == "__main__":
    main()
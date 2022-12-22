import gym
from ai_dm.Environments.gym_envs.gym_problem import GymProblem
from ai_dm.RL.q_learning import q_learning
import ai_dm.base.comp_resources as comp_resources



def test_taxi_q_learning():

    # define the environment
    taxi_env = gym.make("Taxi-v3", render_mode='ansi').env
    taxi_env.reset()
    init_state = taxi_env.encode(0, 3, 4, 1)  # (taxi row, taxi column, passenger index, destination index)
    taxi_row, taxi_col, pass_idx, dest_idx = taxi_env.decode(init_state)
    print(taxi_row)
    taxi_env.unwrapped.s = init_state
    print("State:", init_state)
    print(taxi_env.render())


    # create a wrapper of the environment to the search
    taxi_p = GymProblem(taxi_env, taxi_env.unwrapped.s)

    # perform q_learning




    [best_value, best_node, best_plan, explored_count, ex_terminated] = q_learning(problem=taxi_p,learning_rate=0.9, discount_rate=0.8, epsilon=1.0, decay_rate=0.005, num_episodes=1000,
                   max_steps_per_episode=99, log=False, log_file=None)

    print(best_plan)
    for action_id in best_plan:
        taxi_p.apply_action(action_id)
        taxi_p.env.render()



if __name__ == "__main__":
    test_taxi_q_learning()
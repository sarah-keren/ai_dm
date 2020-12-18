from environments.transforms.environment_transforms import *
from td_agents import *
from mc_agents import *
from environments.gridworld.gridworld_utils import *


def get_agent(agent_name, num_actions, theta, env):
    if agent_name == Q_LEARNING_AGENT:
        return QLearningAgent(num_actions=num_actions, theta=theta, alpha=0.025, gamma=0.9,
                              mapping_fn=lambda x: np.squeeze(np.eye(500)[np.array(x).reshape(-1)]) / 500)
    elif agent_name == SARSA_AGENT:
        return SarsaAgent(num_actions=num_actions, theta=theta, alpha=0.025, gamma=0.9,
                          mapping_fn=lambda x: np.squeeze(np.eye(500)[np.array(x).reshape(-1)]) / 500)
    elif agent_name == MONTE_CARLO_ON_POLICY:
        return MCOnPolicyAgent(env=env, num_actions=num_actions, theta=theta, alpha=0.025, gamma=0.9,
                               mapping_fn=lambda x: np.squeeze(np.eye(500)[np.array(x).reshape(-1)]) / 500)
    elif agent_name == MONTE_CARLO_OFF_POLICY:
        return MCOffPolicyAgent(env=env, num_actions=num_actions, theta=theta, alpha=0.025, gamma=0.9,
                                mapping_fn=lambda x: np.squeeze(np.eye(500)[np.array(x).reshape(-1)]) / 500)
    else:
        print("Not valid agent name")
        return


def get_discrete_environment(env_name, transform_env=False, mapping_class=None):
    if not transform_env:
        if env_name == TAXI:
            return TransformEnvironment(env_name)
        elif env_name == CLIFF_WALKING:
            return TransformEnvironment(env_name)
        elif env_name == MANUAL:
            return get_grid_world()
    elif transform_env:
        if not mapping_class:
            raise Exception("Using a transform environment without mapping_fn")
        return TransformEnvironment(env_name, mapping_class)
    else:
        print("Invalid environment name")


def test_discrete(agent_name, world, transform_env):
    env = get_discrete_environment(world, transform_env, DimReductionMapping)

    num_actions = env.action_space.n
    num_states = env.observation_space.n
    theta = np.random.rand(num_states, num_actions)
    agent = get_agent(agent_name, num_actions, theta, env)
    if world == MANUAL:
        env.set_agent(agent)

    env.reset()
    env.render()

    import train_and_evaluate
    train_and_evaluate.train(env=env, is_env_multiagent=False, agents=[agent], max_episode_len=10000,
                             num_episodes=EPISODES,
                             display=DISPLAY, save_rate=1, agents_save_path="", train_result_path="")
    # train_and_evaluate.evaluate(env=env, is_env_multiagent=False, agents=[agent], max_episode_len=10000,
    #                             num_episodes=1000, display=DISPLAY, save_rate=1, agents_save_path="",
    #                             train_result_path="")


def mapping_func_dim_reduction(action, env, abstract_env, *args):
    new_obs, reward, done, info = env.step(action)
    print(new_obs, reward, done, info)
    not_abstract_state = new_obs
    env_state = list(env.decode(env.s))
    print(env_state)
    env_state[args[0]] = args[1]
    abstract_env.s = abstract_env.encode(*env_state)
    return abstract_env.s, reward, done, info, not_abstract_state


if __name__ == "__main__":
    """ 
    Options for cur_agent_name: 
        - Q_LEARNING_AGENT
        - SARSA_AGENT
        - MONTE_CARLO_ON_POLICY
        - MONTE_CARLO_OFF_POLICY
    Options for env_name:
        - TAXI
        - GRID_WORLD
    """
    cur_agent_name = Q_LEARNING_AGENT
    cur_env_name = TAXI
    with_transform = True
    test_discrete(cur_agent_name, cur_env_name, with_transform)

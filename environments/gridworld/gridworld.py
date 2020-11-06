import sys
from random import random

from environments import mdp, environment
from environments.gridworld import gridworld_utils
from environments.gridworld.gridworld_utils import *


class Gridworld(mdp.MarkovDecisionProcess):
    """
      Gridworld
    """

    def __init__(self, grid):
        # layout
        if type(grid) == type([]):
            grid = _make_grid(grid)
        self.grid = grid

        # parameters
        self.livingReward = 0.0
        self.noise = 0.2

    def set_living_reward(self, reward):
        """
        The (negative) reward for exiting "normal" states.

        Note that in the R+N text, this reward is on entering
        a state and therefore is not clearly part of the state's
        future rewards.
        """
        self.livingReward = reward

    def set_noise(self, noise):
        """
        The probability of moving in an unintended direction.
        """
        self.noise = noise

    def get_possible_actions(self, state):
        """
        Returns list of valid actions for 'state'.

        Note that you can request moves into walls and
        that "exit" states transition to the terminal
        state under the special action "done".
        """
        if state == self.grid.terminalState:
            return ()
        x, y = state
        if type(self.grid[x][y]) == int:
            return ('exit',)
        return 'up', 'left', 'down', 'right'

    def get_states(self):
        """
        Return list of all states.
        """
        # The true terminal state.
        states = [self.grid.terminalState]
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                if self.grid[x][y] != '#':
                    state = (x, y)
                    states.append(state)
        return states

    def get_reward(self, state, action, next_state):
        """
        Get reward for state, action, nextState transition.

        Note that the reward depends only on the state being
        departed (as in the R+N book examples, which more or
        less use this convention).
        """
        if state == self.grid.terminalState:
            return 0.0
        x, y = state
        cell = self.grid[x][y]
        if type(cell) == int or type(cell) == float:
            return cell
        return self.livingReward

    def get_start_state(self):
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                if self.grid[x][y] == 'S':
                    return x, y
        raise Exception('Grid has no start state')

    def is_terminal(self, state):
        """
        Only the TERMINAL_STATE state is *actually* a terminal state.
        The other "exit" states are technically non-terminals with
        a single action "exit" which leads to the true terminal state.
        This convention is to make the grids line up with the examples
        in the R+N textbook.
        """
        return state == self.grid.terminalState

    def get_transition_states_and_probs(self, state, action):
        """
        Returns list of (nextState, prob) pairs
        representing the states reachable
        from 'state' by taking 'action' along
        with their transition probabilities.
        """

        if action not in self.get_possible_actions(state):
            raise Exception("Illegal action!")

        if self.is_terminal(state):
            return []

        x, y = state

        if type(self.grid[x][y]) == int or type(self.grid[x][y]) == float:
            term_state = self.grid.terminalState
            return [(term_state, 1.0)]

        successors = []

        up_state = (self.__isAllowed(y + 1, x) and (x, y + 1)) or state
        left_state = (self.__isAllowed(y, x - 1) and (x - 1, y)) or state
        down_state = (self.__isAllowed(y - 1, x) and (x, y - 1)) or state
        right_state = (self.__isAllowed(y, x + 1) and (x + 1, y)) or state

        if action == 'up' or action == 'down':
            if action == 'up':
                successors.append((up_state, 1 - self.noise))
            else:
                successors.append((down_state, 1 - self.noise))

            mass_left = self.noise
            successors.append((left_state, mass_left / 2.0))
            successors.append((right_state, mass_left / 2.0))

        if action == 'left' or action == 'right':
            if action == 'left':
                successors.append((left_state, 1 - self.noise))
            else:
                successors.append((right_state, 1 - self.noise))

            mass_left = self.noise
            successors.append((up_state, mass_left / 2.0))
            successors.append((down_state, mass_left / 2.0))

        successors = self.__aggregate(successors)

        return successors

    def __aggregate(self, states_and_probs):
        counter = gridworld_utils.Counter()
        for state, prob in states_and_probs:
            counter[state] += prob
        new_states_and_probs = []
        for state, prob in counter.items():
            new_states_and_probs.append((state, prob))
        return new_states_and_probs

    def __isAllowed(self, y, x):
        if y < 0 or y >= self.grid.height:
            return False
        if x < 0 or x >= self.grid.width:
            return False
        return self.grid[x][y] != '#'


class GridworldEnvironment(environment.Environment):

    def __init__(self, grid_world):
        self.gridWorld = grid_world
        self.action_space = ActionSpace()
        self.observation_space = ObservationSpace(grid_world.grid.height * grid_world.grid.width)
        self.reset()
        self.display = get_display(grid_world)
        self.display_callback = None
        self.agent = None
        self.state = None

    def get_current_state(self):
        return self.state

    def get_possible_actions(self, state):
        return self.gridWorld.get_possible_actions(state)

    def step(self, action):
        done = False
        successors = self.gridWorld.get_transition_states_and_probs(self.state, action)
        cur_sum = 0.0
        rand = random.random()
        state = self.get_current_state()
        for nextState, prob in successors:
            cur_sum += prob
            if cur_sum > 1.0:
                raise Exception('Total transition probability more than one; sample failure.')
            if rand < cur_sum:
                reward = self.gridWorld.get_reward(state, action, nextState)
                if nextState == 'TERMINAL_STATE':
                    done = True
                self.state = nextState
                return nextState, reward, done, True
        raise Exception('Total transition probability less than one; sample failure.')

    def reset(self):
        self.state = self.gridWorld.get_start_state()
        return self.state

    def render(self):
        # self.display.displayValues(self.agent, message="VALUES AFTER " + str(ITERATIONS) + " ITERATIONS")
        # self.display.pause()
        self.display.displayQValues(self.agent, message="Q-VALUES AFTER " + str(ITERATIONS) + " ITERATIONS")
        # self.display.pause()

    def set_display(self, display):
        self.display = display

    def set_agent(self, agent):
        self.agent = agent
        terminal_states = self.get_terminal_states()
        agent.set_terminal_states(terminal_states)
        self.display_callback = self.display.displayQValues(self.agent, self.state, "CURRENT Q-VALUES")

    def get_terminal_states(self):
        terminal_states = {}
        for x in range(self.gridWorld.grid.width):
            for y in range(self.gridWorld.grid.height):
                if type(self.gridWorld.grid[x][y]) == int:
                    terminal_states[(x, y)] = True
        return terminal_states


class Grid:
    """
    A 2-dimensional array of immutables backed by a list of lists.  Data is accessed
    via grid[x][y] where (x,y) are cartesian coordinates with x horizontal,
    y vertical and the origin (0,0) in the bottom left corner.

    The __str__ method constructs an output that is oriented appropriately.
    """

    def __init__(self, width, height, initial_value=' '):
        self.width = width
        self.height = height
        self.data = [[initial_value for y in range(height)] for x in range(width)]
        self.terminalState = 'TERMINAL_STATE'

    def __getitem__(self, i):
        return self.data[i]

    def __setitem__(self, key, item):
        self.data[key] = item

    def __eq__(self, other):
        if other is None:
            return False
        return self.data == other.data

    def __hash__(self):
        return hash(self.data)

    def copy(self):
        g = Grid(self.width, self.height)
        g.data = [x[:] for x in self.data]
        return g

    def deep_copy(self):
        return self.copy()

    def shallow_copy(self):
        g = Grid(self.width, self.height)
        g.data = self.data
        return g

    def _get_legacy_text(self):
        t = [[self.data[x][y] for x in range(self.width)] for y in range(self.height)]
        t.reverse()
        return t

    def __str__(self):
        return str(self._get_legacy_text())


def _make_grid(grid_string):
    width, height = len(grid_string[0]), len(grid_string)
    grid = Grid(width, height)
    for ybar, line in enumerate(grid_string):
        y = height - ybar - 1
        for x, el in enumerate(line):
            grid[x][y] = el
    return grid


def getCliffGrid():
    return Gridworld(_make_grid(CLIFF_GRID))


def getCliffGrid2():
    return Gridworld(CLIFF_GRID2)


def getCliffGrid3():
    return Gridworld(_make_grid(CLIFF_GRID3))


def get_discount_grid():
    return Gridworld(DISCOUNT_GRID)


def get_bridge_grid():
    return Gridworld(BRIDGE_GRID)


def get_book_grid():
    return Gridworld(BOOK_GRID)


def get_maze_grid():
    return Gridworld(MAZE_GRID)


def get_user_action(state, action_function):
    """
    Get an action from the user (rather than the agent).

    Used for debugging and lecture demos.
    """
    from environments.gridworld import graphics
    action = None
    while True:
        keys = graphics.graphics_utils.wait_for_keys()
        if 'Up' in keys:
            action = 'up'
        if 'Down' in keys:
            action = 'down'
        if 'Left' in keys:
            action = 'left'
        if 'Right' in keys:
            action = 'right'
        if 'q' in keys:
            sys.exit(0)
        if action is None:
            continue
        break
    actions = action_function(state)
    if action not in actions:
        action = actions[0]
    return action


class RandomAgent:
    def action_callback(self, state):
        return random.choice(mdp.get_possible_actions(state))

    def get_value(self, state):
        return 0.0

    def get_q_value(self, state, action):
        return 0.0

    def get_policy(self, state):
        """NOTE: 'random' is a special policy value; don't use it in your code."""
        return 'random'

    def experience_callback(self, state, action, next_state, reward):
        pass


def print_string(x): print(x)

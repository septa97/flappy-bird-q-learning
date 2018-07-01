import time
import math
import numpy as np

from ple.games.flappybird import FlappyBird
from ple import PLE

class Agent():
    def __init__(self, action_space, grid_size):
        """
        A state is represented by a vector of size 3:
            next_pipe_dist_to_player / grid_size
            (next_pipe_top_y - player_y + 512) / grid_size
            (next_pipe_bottom_y - player_y + 512) / grid_size

        There are only 2 possible actions, to jump or not
        """
        x_dist = math.ceil(350/grid_size)
        pipe_top_y = math.ceil(1024/grid_size)
        pipe_bottom_y = math.ceil(1024/grid_size)

        # The Q-value matrix is a 4-dimensional array, 3 of which are the state representation and the remaining dimension represents the action
        self.Q = np.zeros((x_dist, pipe_top_y, pipe_bottom_y, 2))
        self.action_space = action_space # [119, None] -> Constants established by Pygame Learning Environment. 119 for jump and None for doing nothing
        self.grid_size = grid_size
        self._alpha = 0.1
        self._lambda = 1

    def act(self, p, action):
        reward = p.act(self.action_space[action])
        
        if reward >= 0:
            return 1
        else:
            return -1000

    def get_current_state(self, observation):
        state = np.zeros((3,), dtype=int)

        state[0] = observation['next_pipe_dist_to_player'] // self.grid_size
        state[1] = (observation['next_pipe_top_y'] - observation['player_y'] + 512) // self.grid_size
        state[2] = (observation['next_pipe_bottom_y'] - observation['player_y'] + 512) // self.grid_size

        return state

    def optimal_action(self, state):
        jump = self.Q[state[0], state[1], state[2], 0]
        not_jump = self.Q[state[0], state[1], state[2], 1]

        if jump > not_jump:
            return 0
        else:
            return 1

    def update_Q(self, s, s_prime, reward, action):
        self.Q[s[0], s[1], s[2], action] = (1-self._alpha) * self.Q[s[0], s[1], s[2], action] + self._alpha * (reward + self._lambda * np.max(self.Q[s_prime[0], s_prime[1], s_prime[2]]))


if __name__ == "__main__":
    game = FlappyBird()
    p = PLE(game, fps=30, display_screen=True)
    agent = Agent(action_space=p.getActionSet(), grid_size=10)

    p.init()

    s = agent.get_current_state(game.getGameState())
    episodes = 0
    max_score = 0

    while True:
        # Find the optimal action based on the current state
        max_action = agent.optimal_action(s)

        current_score = p.score()
        max_score = max(current_score, max_score)

        # Perform the optimal action and return the reward
        reward = agent.act(p, max_action)

        # Get the next game state after performing the optimal action
        s_prime = agent.get_current_state(game.getGameState())

        # Update the Q-value matrix
        agent.update_Q(s, s_prime, reward, max_action)
        s = s_prime

        # Uncomment this if you want to see the agent learning in a "normal" speed. Adjust the parameter depending on what speed you want.
        # time.sleep(0.01)

        if p.game_over():
            episodes += 1
            print('Episodes: %s, Current score: %s, Max score: %s' % (episodes, current_score, max_score))
            p.reset_game()


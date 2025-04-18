import time
import logging

import numpy as np
from pyboy import PyBoy
from gymnasium import Env, spaces
from pyboy.utils import WindowEvent

action_names = {
    WindowEvent.PRESS_ARROW_LEFT: "LEFT",
    WindowEvent.PRESS_ARROW_RIGHT: "RIGHT",
    WindowEvent.PRESS_ARROW_DOWN: "DOWN",
    WindowEvent.PRESS_ARROW_UP: "UP",
    WindowEvent.PRESS_BUTTON_A: "A",
    WindowEvent.PRESS_BUTTON_B: "B",
    WindowEvent.PASS: "PASS",
    WindowEvent.PRESS_BUTTON_START: "START",
}

def parse_action(s):
    action = s.strip().upper()
    if action == "LEFT":
        return WindowEvent.PRESS_ARROW_LEFT
    elif action == "RIGHT":
        return WindowEvent.PRESS_ARROW_RIGHT
    elif action == "DOWN":
        return WindowEvent.PRESS_ARROW_DOWN
    elif action == "UP":
        return WindowEvent.PRESS_ARROW_UP
    elif action == "A":
        return WindowEvent.PRESS_BUTTON_A
    elif action == "B":
        return WindowEvent.PRESS_BUTTON_B
    elif action == "PASS":
        return WindowEvent.PASS
    elif action == "START":
        return WindowEvent.PRESS_BUTTON_START
    else:
        raise ValueError("Invalid action: {}".format(action))

class TetrisEnv(Env):
    """
    Defines an environment for managing the game state, the agent's actions, and the
    reward system for the Tetris game.
    """

    def __init__(self, gb_path="", init_state="", speedup=1, action_freq=24, window="SDL2", log_level="ERROR", reward_system=0):
        self.gb_path = gb_path
        self.init_state = init_state
        self.speedup = speedup
        self.action_freq = action_freq
        self.window = window
        logging.basicConfig(level=log_level.upper())
        self.reward_system = reward_system

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            #WindowEvent.PRESS_ARROW_DOWN,
            #WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PASS,
        ]

        self.release_arrow = [
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_UP,
        ]

        self.release_button = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
        ]

        self.sprite_tiles = [i for i in range(120, 140)]
        self.output_shape = (1, 18, 10)
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.board = np.zeros(self.output_shape)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=self.output_shape, dtype=np.uint8
        )
        self.current_score = 0

        self.pyboy = PyBoy(
            self.gb_path,
            debugging=False,
            disable_input=False,
            window_type=self.window,
            hide_window=False,
            # sound=True,
        )

        self.screen = self.pyboy.botsupport_manager().screen()
        self.manager = self.pyboy.botsupport_manager()

        self.pyboy.set_emulation_speed(0 if self.window == "headless" else self.speedup)
        self.reset()
        self.seed_value = None

    def seed(self, seed=None):
        self.seed_value = seed
        np.random.seed(seed)

    def reset(self, seed=None):
        self.seed = seed

        # Load the initial state
        if self.init_state != "":
            with open(self.init_state, "rb") as f:
                self.pyboy.load_state(f)

        observation = self.render()
        self.current_score = self.get_total_score(observation[0])
        self.board = observation
        return observation, {}
    
    def step(self, action):
        self.do_input(self.valid_actions[action])
        observation = self.render()
        if observation[0][0].sum() >= len(observation[0][0]):
            # Game over
            # Pure reward system: No game over penalty
            # Tetris-Gymnasium-like reward system: Game over penalty of -2
            # Custom reward system: Game over penalty of -100
            if self.reward_system == 0:
                return observation, 0, True, False, {}
            elif self.reward_system == 1:
                return observation, -2, True, False, {}
            else:
                return observation, -18, True, False, {}

        # Set reward equal to difference between current and previous score
        total_score = self.get_total_score(observation[0])
        reward = total_score - self.current_score
        
        # Calculate reward based on height changes
        # Include line clears only in Tetris-Gymnasium-like reward system
        if self.reward_system == 1:
            reward = max(reward, 0) # Ignore negative values, only line clears count
                
        #Include step reward in Tetris-Gymnasium-like and Custom reward systems        
        if self.reward_system != 0:
            reward += 0.001
        self.current_score = total_score
        self.board = observation

        logging.debug("Total Score: {}".format(total_score))
        logging.debug("Reward: {}".format(reward))

        return observation, reward, False, False, {}
    
    def render(self):
        # Render the sprite map on the backgound
        background = np.asarray(self.manager.tilemap_background()[2:12, 0:18])
        self.observation = np.where(background == 47, 0, 1)

        # Find all tile indexes for the current tetromino
        sprite_indexes = self.manager.sprite_by_tile_identifier(self.sprite_tiles, on_screen=False)
        for sprite_tiles in sprite_indexes:
            for sprite_idx in sprite_tiles:
                sprite = self.manager.sprite(sprite_idx)
                tile_x = (sprite.x // 8) - 2
                tile_y = sprite.y // 8
                if tile_x < self.output_shape[1] and tile_y < self.output_shape[0]:
                    self.observation[tile_y, tile_x] = 1
        
        # Add a channel dimension to the observation
        self.observation = np.expand_dims(self.observation, axis=0)  # Shape becomes (1, 18, 10)
        logging.debug("Board State:\n{}".format(self.observation))
        return self.observation

    def get_total_score(self, observation):
        # Include score in Pure and Custom reward systems
        score = self.get_score() if (self.reward_system == 0 or self.reward_system == 2) else 0
        logging.debug("Score: {}".format(score))

        # Include board score in Tetris-Gymnasium-like and Custom reward systems
        board_reward = self.get_board_score(observation) if (self.reward_system == 1 or self.reward_system == 2) else 0
        logging.debug("Board Reward: {}".format(board_reward))

        #placement_reward = self.get_placement_score(observation)
        #surface_score = self.get_surface_area(observation) * -1

        #print("Board Reward: {}".format(board_reward))
        #print("Placement Reward: {}".format(placement_reward))
        #print("Surface Score: {}".format(surface_score))

        scores = [
            score,
            board_reward,
            #placement_reward,
            #surface_score,
        ]
        return np.sum(scores)

    def get_score(self):
        raw_value = self.pyboy.get_memory_value(0xC0A0)
        
        # Score is stored in BCD format, convert it to decimal
        # https://datacrystal.tcrf.net/wiki/Tetris_(Game_Boy)/RAM_map

        # Convert the decimal value to a binary string
        bit_string = f"{raw_value:b}"  # Convert to binary without the '0b' prefix

        # Pad the bit string to ensure its length is a multiple of 4
        bit_string = bit_string.zfill((len(bit_string) + 3) // 4 * 4)

        # Split the bit string into 4-bit chunks and interpret as BCD
        bcd_digits = [int(bit_string[i:i + 4], 2) for i in range(0, len(bit_string), 4)]

        # Combine the BCD digits into a single decimal number
        return int("".join(map(str, bcd_digits)))

    
    def get_placement_score(self, board):
        score = 0
        height = self.get_max_height(board)
        for i in range(len(board)):
            diff = np.sum(board[i] - self.board[0][i])
            score += diff * i
        return score
    
    def get_surface_area(self, board):
        area = 0
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == 1:
                    adj = self.get_adjacent(board, i, j)
                    for a in adj:
                        if board[a[0]][a[1]] == 0:
                            area += 1
        return area
    
    def get_board_score(self, board):
        #n = len(board)
        #score_vector = [i / n for i in range(n)]
        #for i in range(len(board)):
        #    current_row = np.sum(board[i]) / len(board[i])
        #    score += current_row * score_vector[i]
        hole_score = self.count_holes(board) * -1
        
        height = self.get_max_height(board)
        width = self.get_max_width(board)
        stack_score = height * -1

        completion_score = 0
        for i in range(len(board)):
            completion = np.sum(board[i]) / len(board[i])
            completion *= i / len(board)
            completion_score += completion

        #print("Holes: {}".format(hole_score))
        #print("Stack: {}".format(stack_score))
        #print("Completion: {}".format(completion_score))
        
        if self.reward_system == 1:
            return stack_score # Tetris-Gymnasium-like reward system
        elif self.reward_system == 2:
            return hole_score # Custom reward system
        else:
            return 0 # Shouldn't get called, but just in case.

        #return stack_score
        #return hole_score
        #return completion_score
        #return hole_score + stack_score
        #return hole_score + stack_score + completion_score
    
    # def get_max_height(self, board):
    #     return np.max(np.sum(board, axis=0))
    
    def get_max_height(self, board):
        # Ignore the top row(s) when calculating height
        rows_with_blocks = np.any(board[1:], axis=1)  # Skip the first row
        if np.any(rows_with_blocks):
            return len(board) - np.argmax(rows_with_blocks) - 1  # Adjust for skipped row
        return 0
    
    def get_max_width(self, board):
        return np.max(np.sum(board, axis=1))
    
    def count_holes(self, board):
        """
        Count all the "holes" in the board
        """
        holes = 0
        for i in range(len(board)):
            for j in range(len(board[i])):
                holes += self.is_hole(board, i, j)
        return holes
    
    def is_hole(self, board, x, y):
        """
        Check if a given coordinate is a hole
        """
        if board[x][y] == 1:
            return False
        for adj in self.get_adjacent(board, x, y):
            if board[adj[0]][adj[1]] == 0:
                return False
        return True
    
    def get_adjacent(self, board, x, y):
        """
        Get all the adjacent coordinates for a given coordinate
        """
        adjacent = []
        shape = board.shape
        if x > 0:
            adjacent.append((x - 1, y))
        if x < shape[0] - 1:
            adjacent.append((x + 1, y))
        if y > 0:
            adjacent.append((x, y - 1))
        if y < shape[1] - 1:
            adjacent.append((x, y + 1))
        return adjacent

    
    def do_input(self, action):
        # Press and release the button to simulate human input
        self.pyboy.send_input(action)
        for i in range(self.action_freq):
            if i == 8:
                if action < 4:
                    self.pyboy.send_input(self.release_arrow[action])
                if action > 3 and action < 6: 
                    self.pyboy.send_input(self.release_button[action - 4])
                if action == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            self.pyboy.tick()
        logging.debug("Action: {}".format(action_names[action]))

    def save_state(self, dest=""):
        if dest == "":
            dest = time.strftime("%Y%m%d-%H%M%S.save")

        with open(dest, "wb") as f:
            self.pyboy.save_state(f)

    def load_state(self, src):
        with open(src, "rb") as f:
            self.pyboy.load_state(f)
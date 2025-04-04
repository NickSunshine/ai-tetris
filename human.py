import argparse
import os
import time
from tetris_env import TetrisEnv

def parse_args():
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--rom", type=str, default="roms/tetris.gb", help="Path to the ROM file.")
    parser.add_argument("--init", type=str, default="states/init.state", help="Path to the initial state.")
    parser.add_argument("--speedup", type=int, default=1, help="Speedup factor for human rendering.")
    parser.add_argument("--freq", type=int, default=24, help="Action frequency.")
    parser.add_argument("--log-level", type=str, default="ERROR", help="Logging level.")
    return parser.parse_args()

def play_human(args):
    # Configure the environment for human rendering
    env = TetrisEnv(
        gb_path=args.rom,
        action_freq=args.freq,
        speedup=args.speedup,
        init_state=args.init,
        log_level=args.log_level,
        window="SDL2"  # Enable human rendering
    )

    # Reset the environment
    obs, _ = env.reset()
    done = False

    print("Playing Tetris with random actions. Close the window to exit.")

    try:
        while not done:
            # Take a random action
            action = env.action_space.sample()
            obs, reward, done, _, _ = env.step(action)

            # Render the environment (human rendering is automatic with PyBoy)
            #time.sleep(1 / args.freq)  # Control the speed of the game
    except KeyboardInterrupt:
        print("Exiting the game.")
    finally:
        env.close()

if __name__ == "__main__":
    args = parse_args()
    play_human(args)
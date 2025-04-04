import argparse
import os
import time
import random  # Import random for generating random seeds
from tetris_env import TetrisEnv
from stable_baselines3 import PPO  # Import PPO for loading the model

def parse_args():
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--rom", type=str, default="roms/tetris.gb", help="Path to the ROM file.")
    parser.add_argument("--init", type=str, default="states/init.state", help="Path to the initial state.")
    parser.add_argument("--speedup", type=int, default=1, help="Speedup factor for human rendering.")
    parser.add_argument("--freq", type=int, default=24, help="Action frequency.")
    parser.add_argument("--log-level", type=str, default="ERROR", help="Logging level.")
    parser.add_argument("--model", type=str, default=None, help="Path to the model file to load.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the environment.")
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

    # Load the model if provided
    model = None
    if args.model:
        print(f"Loading model from {args.model}...")
        model = PPO.load(args.model)

    # Set the random seed
    seed = args.seed if args.seed is not None else random.randint(0, 10000)
    print(f"Using random seed: {seed}")
    env.reset(seed=seed)

    done = False

    print("Playing Tetris. Close the window to exit.")

    try:
        while not done:
            # Use the model for action prediction if available, otherwise sample randomly
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            # Step the environment
            obs, reward, done, _, _ = env.step(action)

            # Print the reward to stdout
            print(f"Reward: {reward}")

            # Render the environment (human rendering is automatic with PyBoy)
            time.sleep(1 / args.freq)  # Control the speed of the game
    except KeyboardInterrupt:
        print("Exiting the game.")
    finally:
        env.close()

if __name__ == "__main__":
    args = parse_args()
    play_human(args)
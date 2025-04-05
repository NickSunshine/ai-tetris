import argparse
import time
from tetris_env import TetrisEnv
from stable_baselines3 import PPO
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rom", type=str, default="roms/tetris.gb", help="Path to the ROM file.")
    parser.add_argument("--init", type=str, default="states/init.state", help="Path to the initial state.")
    parser.add_argument("--speedup", type=int, default=1, help="Speedup factor for human rendering.")
    parser.add_argument("--freq", type=int, default=24, help="Action frequency.")
    parser.add_argument("--log-level", type=str, default="ERROR", help="Logging level.")
    parser.add_argument("--model", type=str, default=None, help="Path to the model file to load.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the environment.")
    parser.add_argument("--render-mode", type=str, choices=["SDL2", "headless"], default="headless",
                        help="Render mode for the environment: 'SDL2' for visual rendering, 'headless' for no rendering.")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs to evaluate the model.")
    return parser.parse_args()

def eval(args):

    # Configure the environment for Tetris
    env = TetrisEnv(
        gb_path=args.rom,
        action_freq=args.freq,
        speedup=args.speedup,
        init_state=args.init,
        log_level=args.log_level,
        window=args.render_mode
    )

    # Load a model if specified, otherwise use a random policy
    model = None
    if args.model:
        print(f"Loading model from {args.model}...")
        model = PPO.load(args.model)

    print(f"Evaluating Tetris for {args.runs} runs. Close the window to exit.")

    # Perform multiple runs
    for run in range(args.runs):
        # Set a random seed for each run
        seed = args.seed if args.seed is not None else np.random.randint(0, 100000)
        obs, _ = env.reset(seed=seed)
        terminated = False
        steps = 0

        while not terminated:
            # Use the model for action prediction if available, otherwise sample randomly
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, _, _ = env.step(action)

            # Control the speed of the game during visual rendering
            #if args.render_mode == "SDL2":
            #    time.sleep(1 / args.freq)

            steps += 1

        # Print the results of the run
        print(f"Run {run + 1}/{args.runs}: Seed: {seed}, Steps: {steps}, Score: {env.get_score()}")

    env.close()

if __name__ == "__main__":
    eval(parse_args())
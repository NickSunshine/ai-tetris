import argparse
import time
from tetris_env import TetrisEnv
from stable_baselines3 import PPO
import numpy as np
import os  # For creating directories
import matplotlib.pyplot as plt  # For plotting
from datetime import datetime  # For generating timestamps

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

    # Lists to store steps and scores for each run
    steps_list = []
    scores_list = []

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
            if args.render_mode == "SDL2":
                time.sleep(1 / args.freq)

            steps += 1

        # Record the results of the run
        steps_list.append(steps)
        scores_list.append(env.get_score())

        # Print the results of the run
        print(f"Run {run + 1}/{args.runs}: Seed: {seed}, Steps: {steps}, Score: {env.get_score()}")

    # Close the environment
    env.close()

    # Compute and display summary statistics
    print("\nSummary Statistics:")
    print(f"N: {len(steps_list)}")  # Number of runs
    print(f"Steps - Min: {np.min(steps_list)}, Max: {np.max(steps_list)}, Avg: {np.mean(steps_list):.2f}, Std: {np.std(steps_list):.2f}")
    print(f"Score - Min: {np.min(scores_list)}, Max: {np.max(scores_list)}, Avg: {np.mean(scores_list):.2f}, Std: {np.std(scores_list):.2f}")

    # Create directories for plots if they don't exist
    histogram_dir = os.path.join("plots", "histograms")
    os.makedirs(histogram_dir, exist_ok=True)

    # Generate a timestamp for the filenames (evaluation completion time)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Determine the policy name from the model path or use "Random"
    if args.model:
        # Extract the policy name from the model path (e.g., "CnnPolicy" or "MlpPolicy")
        policy_name = os.path.basename(os.path.dirname(args.model))
        # Extract timesteps from the model filename (e.g., "4096" from "20250404-222051_4096.zip")
        timesteps = os.path.basename(args.model).split("_")[-1].replace(".zip", "")
    else:
        policy_name = "Random"
        timesteps = "NA"  # Not applicable for random gameplay

    # Clean up the policy name for filenames
    policy_name_clean = policy_name.replace("/", "_").replace("\\", "_")

    # Plot and save the histogram of steps
    plt.figure()
    plt.hist(steps_list, bins=10, color='blue', edgecolor='black', alpha=0.7)
    plt.title(f"Histogram of Steps ({policy_name})")
    plt.xlabel("Steps")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    histogram_path = os.path.join(histogram_dir, f"steps_histogram_{policy_name_clean}_{timesteps}_{timestamp}.png")
    plt.savefig(histogram_path)
    plt.close()
    print(f"Histogram of steps saved to {histogram_path}")

    # Plot and save the histogram of scores
    plt.figure()
    plt.hist(scores_list, bins=10, color='green', edgecolor='black', alpha=0.7)
    plt.title(f"Histogram of Scores ({policy_name})")
    plt.xlabel("Scores")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    score_histogram_path = os.path.join(histogram_dir, f"scores_histogram_{policy_name_clean}_{timesteps}_{timestamp}.png")
    plt.savefig(score_histogram_path)
    plt.close()
    print(f"Histogram of scores saved to {score_histogram_path}")

if __name__ == "__main__":
    eval(parse_args())
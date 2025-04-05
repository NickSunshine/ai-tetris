import argparse
import time
from tetris_env import TetrisEnv
from stable_baselines3 import PPO
import numpy as np
import os  # For creating directories
import matplotlib.pyplot as plt  # For plotting
from datetime import datetime  # For generating timestamps
import logging  # For logging

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
    parser.add_argument("--plot", action="store_true", help="Generate plots if this flag is provided.")
    return parser.parse_args()

def eval(args):
    # Create directories for logs if they don't exist
    log_dir = os.path.join("logs", "eval")
    os.makedirs(log_dir, exist_ok=True)

    # Generate a timestamp for the log filename
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Determine the policy name for the log filename
    if args.model:
        policy_name = os.path.basename(os.path.dirname(args.model))
        timesteps = os.path.basename(args.model).split("_")[-1].replace(".zip", "")
    else:
        policy_name = "Random"
        timesteps = "NA"

    # Clean up the policy name for filenames
    policy_name_clean = policy_name.replace("/", "_").replace("\\", "_")

    # Configure logging
    log_filename = os.path.join(log_dir, f"evaluation_{policy_name_clean}_{timesteps}_{timestamp}.log")
    logging.basicConfig(
        filename=log_filename,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )

    # Add a stream handler to also log to stdout
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(console_handler)

    # Log the start of evaluation
    logging.info(f"Starting evaluation for policy: {policy_name}, timesteps: {timesteps}, runs: {args.runs} ")
    
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

    # Lists to store steps, scores, and cumulative rewards for each run
    steps_list = []
    scores_list = []
    cumulative_reward_list = []  # New list to store cumulative rewards

    # Perform multiple runs
    for run in range(args.runs):
        # Set a random seed for each run
        seed = args.seed if args.seed is not None else np.random.randint(0, 100000)
        obs, _ = env.reset(seed=seed)
        terminated = False
        steps = 0
        cumulative_reward = 0  # Initialize cumulative reward for the run

        while not terminated:
            # Use the model for action prediction if available, otherwise sample randomly
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, _, _ = env.step(action)

            # Accumulate the reward
            cumulative_reward += reward

            # Control the speed of the game during visual rendering
            if args.render_mode == "SDL2":
                time.sleep(1 / args.freq)

            steps += 1

        # Record the results of the run
        steps_list.append(steps)
        scores_list.append(env.get_score())
        cumulative_reward_list.append(cumulative_reward)  # Add cumulative reward to the list

        # Log the results of the run
        logging.info(f"Run {run + 1}/{args.runs}: Seed: {seed}, Steps: {steps}, Score: {env.get_score()}, Cumulative Reward: {cumulative_reward}")

    # Close the environment
    env.close()

    # Compute and display summary statistics
    logging.info("Summary Statistics:")
    logging.info(f"N: {len(steps_list)}")  # Number of runs
    logging.info(f"Steps - Min: {np.min(steps_list)}, Max: {np.max(steps_list)}, Avg: {np.mean(steps_list):.2f}, Std: {np.std(steps_list):.2f}")
    logging.info(f"Score - Min: {np.min(scores_list)}, Max: {np.max(scores_list)}, Avg: {np.mean(scores_list):.2f}, Std: {np.std(scores_list):.2f}")
    logging.info(f"Cumulative Reward - Min: {np.min(cumulative_reward_list)}, Max: {np.max(cumulative_reward_list)}, Avg: {np.mean(cumulative_reward_list):.2f}, Std: {np.std(cumulative_reward_list):.2f}")

    if args.plot:
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
        steps_histogram_path = os.path.join(histogram_dir, f"steps_histogram_{policy_name_clean}_{timesteps}_{timestamp}.png")
        plt.savefig(steps_histogram_path)
        plt.close()
        logging.info(f"Histogram of steps saved to {steps_histogram_path}")

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
        logging.info(f"Histogram of scores saved to {score_histogram_path}")

        # Plot and save the line plot of steps
        plt.figure()
        plt.plot(range(1, len(steps_list) + 1), steps_list, marker='o', color='blue', linestyle='-', linewidth=2)
        plt.title(f"Steps per Run ({policy_name})")
        plt.xlabel("Run Number")
        plt.ylabel("Steps")
        plt.grid(axis='both', linestyle='--', alpha=0.7)
        steps_line_plot_path = os.path.join(histogram_dir, f"steps_lineplot_{policy_name_clean}_{timesteps}_{timestamp}.png")
        plt.savefig(steps_line_plot_path)
        plt.close()
        logging.info(f"Line plot of steps saved to {steps_line_plot_path}")

        # Plot and save the line plot of scores
        plt.figure()
        plt.plot(range(1, len(scores_list) + 1), scores_list, marker='o', color='green', linestyle='-', linewidth=2)
        plt.title(f"Scores per Run ({policy_name})")
        plt.xlabel("Run Number")
        plt.ylabel("Score")
        plt.grid(axis='both', linestyle='--', alpha=0.7)
        score_line_plot_path = os.path.join(histogram_dir, f"scores_lineplot_{policy_name_clean}_{timesteps}_{timestamp}.png")
        plt.savefig(score_line_plot_path)
        plt.close()
        logging.info(f"Line plot of scores saved to {score_line_plot_path}")

        # Plot and save the boxplot of steps
        plt.figure()
        plt.boxplot(steps_list, vert=True, patch_artist=True, boxprops=dict(facecolor='blue', color='black'), medianprops=dict(color='red'))
        plt.title(f"Boxplot of Steps ({policy_name})")
        plt.ylabel("Steps")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        steps_boxplot_path = os.path.join(histogram_dir, f"steps_boxplot_{policy_name_clean}_{timesteps}_{timestamp}.png")
        plt.savefig(steps_boxplot_path)
        plt.close()
        logging.info(f"Boxplot of steps saved to {steps_boxplot_path}")

        # Plot and save the boxplot of scores
        plt.figure()
        plt.boxplot(scores_list, vert=True, patch_artist=True, boxprops=dict(facecolor='green', color='black'), medianprops=dict(color='red'))
        plt.title(f"Boxplot of Scores ({policy_name})")
        plt.ylabel("Scores")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        scores_boxplot_path = os.path.join(histogram_dir, f"scores_boxplot_{policy_name_clean}_{timesteps}_{timestamp}.png")
        plt.savefig(scores_boxplot_path)
        plt.close()
        logging.info(f"Boxplot of scores saved to {scores_boxplot_path}")

        # Plot and save the scatterplot of steps vs. scores
        plt.figure()
        plt.scatter(steps_list, scores_list, color='purple', alpha=0.7, edgecolor='black')
        plt.title(f"Scatterplot of Steps vs. Scores ({policy_name})")
        plt.xlabel("Steps")
        plt.ylabel("Scores")
        plt.grid(axis='both', linestyle='--', alpha=0.7)
        scatter_plot_path = os.path.join(histogram_dir, f"scatterplot_steps_vs_scores_{policy_name_clean}_{timesteps}_{timestamp}.png")
        plt.savefig(scatter_plot_path)
        plt.close()
        logging.info(f"Scatterplot of steps vs. scores saved to {scatter_plot_path}")

if __name__ == "__main__":
    eval(parse_args())
import argparse
import os
import json
from datetime import datetime
from ulid import ULID
from stable_baselines3 import PPO
from stable_baselines3.common.logger import Logger, make_output_format

from agent import AgentTrainer
from tensorboard_writer import TensorboardWriter
from tetris_env import TetrisEnv
from custom_cnn import CustomCNN

def parse_args():
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--rom", type=str, default="roms/tetris.gb", help="Path to the ROM file.")
    parser.add_argument("--init", type=str, default="states/init.state", help="Path to the initial state.")
    parser.add_argument("--speedup", type=int, default=5, help="Speedup factor.")
    parser.add_argument("--freq", type=int, default=24, help="Action frequency.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for training.")
    parser.add_argument("--steps", type=int, default=2048, help="Number of steps per run.")
    parser.add_argument("--runs", type=int, default=1000, help="Number of runs for training.")
    parser.add_argument("--log-stdout", type=bool, default=True, help="Log session events to stdout.")
    parser.add_argument("--log-level", type=str, default="ERROR", help="Logging level.")
    parser.add_argument("--load_latest_model", action="store_true", help="Load the most recent model from the models directory to continue training.")
    parser.add_argument("--policy", type=str, choices=["MlpPolicy", "CnnPolicy"], default="MlpPolicy", help="Policy type to use for training (MlpPolicy or CnnPolicy).")
    return parser.parse_args()

def train(args):
    # Set the model directory based on the policy
    model_dir = os.path.join("models", args.policy)
    os.makedirs(model_dir, exist_ok=True)

    # Ensure directories exist for logging
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", "tensorboard", f"{args.policy}", f"{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    # Create the writer
    writer = TensorboardWriter(log_dir=log_dir)

    # Configure the environment
    agent_id = ULID()
    env = TetrisEnv(
        gb_path=args.rom,
        action_freq=args.freq,
        speedup=args.speedup,
        init_state=args.init,
        log_level=args.log_level,
        window="headless"
    )

    # Load metrics and the latest model if requested
    metrics_file = os.path.join(model_dir, "training_metrics.json")
    model = None
    total_timesteps = 0
    total_duration = 0

    if args.load_latest_model:
        # Load metrics
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
                total_timesteps = metrics.get("total_timesteps", 0)
                total_duration = metrics.get("total_duration", 0)
                latest_model_path = metrics.get("last_saved_model")
                if latest_model_path and os.path.exists(latest_model_path):
                    print(f"Loading the latest model: {latest_model_path}")
                    model = PPO.load(latest_model_path, env=env)
                else:
                    print("No valid model found in metrics. Training a new model from scratch.")
        else:
            print("No metrics file found. Training a new model from scratch.")

    # If no model was loaded, create a new one
    if model is None:
        print("Training a new model from scratch...")
        policy_kwargs = None

        if args.policy == "CnnPolicy":
            policy_kwargs = dict(
                features_extractor_class=CustomCNN,
                features_extractor_kwargs=dict(features_dim=256),
                normalize_images=False  # Disable image normalization since your input is already normalized
            )

        model = PPO(
            policy=args.policy,
            env=env,
            verbose=1,
            n_steps=args.steps,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            gamma=args.gamma,
            policy_kwargs=policy_kwargs if args.policy == "CnnPolicy" else None
        )

    # Set logging outputs
    output_formats = []
    if args.log_stdout:
        output_formats.append(make_output_format("stdout", "runs"))
    output_formats.append(writer)
    model.set_logger(Logger(None, output_formats=output_formats))

    # Train the model
    trainer = AgentTrainer(writer=writer, model_dir=model_dir, agent_id=agent_id)
    trainer.train(model, runs=args.runs, total_timesteps=total_timesteps, total_duration=total_duration)

    # Close the writer
    writer.close()

if __name__ == "__main__":
    train(parse_args())
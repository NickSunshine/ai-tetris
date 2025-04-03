import asyncio
import argparse

from ulid import ULID
from stable_baselines3 import PPO
from pyensign.ensign import Ensign
from stable_baselines3.common.logger import Logger, make_output_format

from agent import AgentTrainer
from ensign_writer import EnsignWriter
from tensorboard_writer import TensorboardWriter
from tetris_env import TetrisEnv

from writer_factory import get_writer

def parse_args():
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--creds", type=str, default="ensign_creds.json", help="Path to your Ensign credentials file.")
    parser.add_argument("--rom", type=str, default="roms/tetris.gb", help="Path to the ROM file.")
    parser.add_argument("--init", type=str, default="states/init.state", help="Path to the initial state.")
    parser.add_argument("--speedup", type=int, default=5, help="Speedup factor.")
    parser.add_argument("--freq", type=int, default=24, help="Action frequency.")
    parser.add_argument("--policy", type=str, default="MlpPolicy", help="Model policy to use.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for training.")
    parser.add_argument("--steps", type=int, default=2048, help="Number of steps for the model to learn.")
    parser.add_argument("--sessions", type=int, default=40, help="Number of training sessions.")
    parser.add_argument("--runs", type=int, default=4, help="Number of runs per session.")
    parser.add_argument("--session-topic", type=str, default="agent-sessions", help="Topic for publishing session events.")
    parser.add_argument("--model-topic", type=str, default="agent-models", help="Topic for publishing model events.")
    parser.add_argument("--log-stdout", type=bool, default=True, help="Log session events to stdout.")
    parser.add_argument("--log-level", type=str, default="ERROR", help="Logging level.")
    return parser.parse_args()

async def train(args):
    # Create the writer using the factory
    writer = get_writer()

    # Create Ensign client if using EnsignWriter
    ensign = None
    if isinstance(writer, EnsignWriter):
        ensign = Ensign(cred_path=args.creds)

    # Configure the environment, model, and trainer
    agent_id = ULID()
    env = TetrisEnv(
        gb_path=args.rom,
        action_freq=args.freq,
        speedup=args.speedup,
        init_state=args.init,
        log_level=args.log_level,
        window="headless"
    )
    trainer = AgentTrainer(
        ensign=ensign,
        model_topic=args.model_topic,
        agent_id=agent_id
    )
    model = PPO(
        args.policy,
        env,
        verbose=1,
        n_steps=args.steps,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        gamma=args.gamma
    )

    # Set logging outputs
    output_formats = []
    if args.log_stdout:
        output_formats.append(make_output_format("stdout", "sessions"))
    if isinstance(writer, EnsignWriter) or isinstance(writer, TensorboardWriter):
        output_formats.append(writer)
    model.set_logger(Logger(None, output_formats=output_formats))

    # Train the model
    await trainer.train(model, sessions=args.sessions, runs_per_session=args.runs)

    # Save the model locally
    model.save("models/saved_model.pth")

    # Close the writer
    writer.close()

    # Close the Ensign client if used
    if ensign:
        await ensign.close()


if __name__ == "__main__":
    asyncio.run(train(parse_args()))
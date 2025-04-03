import io
import os
import asyncio
from datetime import datetime

from ulid import ULID
from pyensign.events import Event
from base_writer import BaseWriter

class AgentTrainer:
    """
    AgentTrainer can train and evaluate an agent for a reinforcement learning task.
    """

    def __init__(self, writer: BaseWriter, ensign=None, model_topic="agent-models", model_dir="", agent_id=ULID()):
        self.writer = writer
        self.ensign = ensign
        self.model_topic = model_topic
        self.agent_id = agent_id
        self.model_dir = model_dir

    async def train(self, model, sessions=40, runs_per_session=4, model_version="0.1.0"):
        """
        Train the agent for the specified number of steps.
        """

        model_name = model.__class__.__name__
        policy_name = model.policy.__class__.__name__

        if self.ensign:
            await self.ensign.ensure_topic_exists(self.model_topic)

        if self.model_dir:
            os.makedirs(self.model_dir, exist_ok=True)

        # Train for the number of sessions
        for session in range(sessions):
            print(f"Training session {session + 1}/{sessions} for {model_name} with policy {policy_name}...")
            session_start = datetime.now()
            model.learn(total_timesteps=model.n_steps * runs_per_session)
            session_end = datetime.now()
            duration = session_end - session_start

            # Ensure that async loggers have a chance to run
            await asyncio.sleep(5)

            # Log training metrics using the writer
            self.writer.write(
                {
                    "session": session + 1,
                    "train_duration_seconds": duration.total_seconds(),
                    "total_timesteps": (session + 1) * runs_per_session * model.n_steps,
                },
                key_excluded=[],
                step=session + 1,
            )
            print(f"Total timesteps so far: {(session + 1) * runs_per_session * model.n_steps}")
            # Save the model
            if self.ensign:
                buffer = io.BytesIO()
                model.save(buffer)
                model_event = Event(
                    buffer.getvalue(),
                    "application/octet-stream",
                    schema_name=model_name,
                    schema_version=model_version,
                    meta={
                        "agent_id": str(self.agent_id),
                        "model": model_name,
                        "policy": policy_name,
                        "trained_at": session_end.isoformat(),
                        "train_seconds": str(duration.total_seconds()),
                    },
                )
                await self.ensign.publish(self.model_topic, model_event)

            if self.model_dir:
                model.save(
                    os.path.join(
                        self.model_dir,
                        "{}_{}.zip".format(model_name, session_end.strftime("%Y%m%d-%H%M%S")),
                    )
                )

        if self.ensign:
            await self.ensign.flush()

    async def eval(self, eval_topic="eval-agent", model_version="latest"):
        """
        Evaluate the agent in an independent testing environment using the specified
        model version or the latest model.
        """

        if self.ensign:
            await self.ensign.ensure_topic_exists(eval_topic)

    async def run(self, model_version="latest"):
        """
        Run the agent in "demo" mode using the model version.
        """

        pass

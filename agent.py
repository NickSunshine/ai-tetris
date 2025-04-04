import os
from datetime import datetime
from ulid import ULID

class AgentTrainer:
    """
    AgentTrainer can train and evaluate an agent for a reinforcement learning task.
    """

    def __init__(self, writer, model_dir="", agent_id=ULID()):
        self.writer = writer
        self.agent_id = agent_id
        self.model_dir = model_dir

    def train(self, model, sessions=40, runs_per_session=4, total_timesteps=0):
        """
        Train the agent for the specified number of steps.
        """

        model_name = model.__class__.__name__
        policy_name = model.policy.__class__.__name__

        if self.model_dir:
            os.makedirs(self.model_dir, exist_ok=True)

        # Train for the number of sessions
        for session in range(sessions):
            print(f"Training session {session + 1}/{sessions} for {model_name} with policy {policy_name}...")
            session_start = datetime.now()
            model.learn(total_timesteps=model.n_steps * runs_per_session)
            session_end = datetime.now()
            duration = session_end - session_start

            # Update total timesteps
            total_timesteps += model.n_steps * runs_per_session

            # Log training metrics using the writer
            self.writer.write(
                {
                    "session_number": session + 1,
                    "session_train_duration_seconds": duration.total_seconds(),
                    "overall_total_timesteps": (session + 1) * runs_per_session * model.n_steps,
                },
                key_excluded=[],
                step=session + 1,
            )
            print(f"Total timesteps so far: {total_timesteps}")

            # Save the model after each session if a model directory is provided
            if self.model_dir:
                model.save(
                    os.path.join(
                        self.model_dir,
                        "{}_{}.zip".format(model_name, session_end.strftime("%Y%m%d-%H%M%S")),
                    )
                )
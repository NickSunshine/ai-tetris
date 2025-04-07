import os
from datetime import datetime
import json
from ulid import ULID

class AgentTrainer:
    """
    AgentTrainer can train and evaluate an agent for a reinforcement learning task.
    """

    def __init__(self, writer, model_dir="", agent_id=ULID()):
        self.writer = writer
        self.agent_id = agent_id
        self.model_dir = model_dir

    def train(self, model, runs, total_timesteps, total_duration):
        # Generate a single timestamp at the beginning of training
        training_start_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Calculate the interval for saving every 10% of runs
        save_interval = max(1, runs // 10)  # Ensure at least one save if runs < 10

        for run in range(1, runs + 1):
            # Start timing the run
            start_time = datetime.now()

            # Train the model for the specified number of steps
            model.learn(total_timesteps=model.n_steps, reset_num_timesteps=False)

            # Update total timesteps
            total_timesteps += model.n_steps

            # Save the model based on the conditions
            if run == 1 or run == runs or run % save_interval == 0:
                model_path = os.path.join(
                    self.model_dir,
                    f"{training_start_timestamp}_{total_timesteps}.zip"
                )
                model.save(model_path)
                print(f"Model saved to {model_path} after run {run}.")

            # Calculate run duration
            run_duration = (datetime.now() - start_time).total_seconds()

            # Update total duration
            total_duration += run_duration

            # Log progress with detailed run duration
            print(f"Run {run}/{runs} completed.")
            print(f"Run timesteps: {model.n_steps}. Total timesteps: {total_timesteps}. Run duration: {run_duration:.2f} s. Total duration: {total_duration:.2f} s.")

            # Write metrics to TensorBoard
            self.writer.write(
                {
                    "custom/total_timesteps": total_timesteps,
                    "custom/total_duration": total_duration,
                    "custom/run_number": run,
                    "custom/run_duration": run_duration
                },
                key_excluded=[],
                step=run * model.n_steps  # Use the current training's elapsed steps as the X-axis value
            )

        # Save metrics (excluding overall_total_duration)
        metrics = {
            "total_timesteps": total_timesteps,
            "total_duration": total_duration,
            "last_saved_model": model_path
        }
        metrics_file = os.path.join(self.model_dir, "training_metrics.json")
        try:
            with open(metrics_file, "w") as f:
                json.dump(metrics, f)
        except Exception as e:
            print(f"Error saving metrics to {metrics_file}: {e}")
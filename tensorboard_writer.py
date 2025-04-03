import os
from stable_baselines3.common.logger import KVWriter
from torch.utils.tensorboard import SummaryWriter

class TensorboardWriter(KVWriter):
    """
    TensorboardWriter subclasses both BaseWriter and KVWriter to write key-value pairs to TensorBoard.
    """

    def __init__(self, log_dir="logs/tensorboard"):
        super().__init__()
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def write(self, key_values, key_excluded, step=0):
        """
        Write the key-value pairs to TensorBoard.
        """
        #print(f"Logging to TensorBoard: {key_values}")  # Debugging
        #print(f"Excluded keys before modification: {key_excluded}")  # Debugging

        # Remove specific metrics from key_excluded
        metrics_to_include = [
            "rollout/ep_len_mean",
            "rollout/ep_rew_mean",
            "time/fps",
            "time/iterations",
            "time/time_elapsed",
            "time/total_timesteps",
            "train/approx_kl",
            "train/clip_fraction",
            "train/clip_range",
            "train/entropy_loss",
            "train/explained_variance",
            "train/learning_rate",
            "train/loss",
            "train/n_updates",
            "train/policy_gradient_loss",
            "train/value_loss"
        ]

        key_excluded = [key for key in key_excluded if key not in metrics_to_include]
        #print(f"Excluded keys after modification: {key_excluded}")  # Debugging

        # Log all key-value pairs dynamically
        for key, value in key_values.items():
            if key not in key_excluded:
                try:
                    self.writer.add_scalar(key, value, step)
                except ModuleNotFoundError as e:
                    if "caffe2" in str(e):
                        print(f"Warning: {e} (skipping this metric) key: {key}, value: {value}, step: {step}")

    def close(self):
        """
        Close the TensorBoard writer.
        """
        self.writer.close()
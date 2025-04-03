import os
from stable_baselines3.common.logger import KVWriter
from torch.utils.tensorboard import SummaryWriter

class TensorboardWriter(KVWriter):
    """
    TensorboardWriter subclasses the Stable Baselines3 KVWriter class to write key-value pairs to TensorBoard.
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
        for key, value in key_values.items():
            if key not in key_excluded:
                self.writer.add_scalar(key, value, step)

    def close(self):
        """
        Close the TensorBoard writer.
        """
        self.writer.close()
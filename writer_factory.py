import json
from tensorboard_writer import TensorboardWriter

def get_writer(config_path="config.json"):
    """
    Factory function to create a writer based on the configuration.
    """
    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    writer_type = config.get("writer_type", "tensorboard").lower()
    log_dir = config.get("log_dir", "logs/tensorboard")

    if writer_type == "tensorboard":
        return TensorboardWriter(log_dir=log_dir)
    else:
        raise ValueError(f"Unsupported writer type: {writer_type}")
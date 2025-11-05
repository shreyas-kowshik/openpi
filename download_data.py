from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_libero")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# breakpoint()

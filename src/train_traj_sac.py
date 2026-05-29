from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from franka_traj_env import FrankaCircleEnv
import os


class SaveBufferCallback(BaseCallback):
    """Saving replay buffer """
    def __init__(self, save_path, save_freq, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path,
                                f"buffer_{self.n_calls}")
            self.model.save_replay_buffer(path)
        return True


env = Monitor(FrankaCircleEnv(
    obs_noise_std = 0.005,
    action_delay  = 1,
    trajectory    = "figure8",
    fixed_centre  = [0.5, 0.0, 0.5],
    fixed_radius  = 0.4
))

model = SAC(
    "MlpPolicy",
    env,
    learning_rate   = 3e-4,
    buffer_size     = 200_000,
    learning_starts = 5_000,
    batch_size      = 256,
    tau             = 0.005,
    gamma           = 0.99,
    train_freq      = 1,
    gradient_steps  = 1,
    verbose         = 1,
    device          = "cpu",
    policy_kwargs   = dict(net_arch=[256, 256])
)

os.makedirs("models/checkpoints", exist_ok=True)

checkpoint_cb   = CheckpointCallback(
    save_freq   = 100_000,
    save_path   = "models/checkpoints/",
    name_prefix = "sac_fig8_v5"
)
buffer_cb = SaveBufferCallback(
    save_path = "models/checkpoints/",
    save_freq = 100_000
)

model.learn(
    total_timesteps = 1_000_000,
    callback        = [checkpoint_cb, buffer_cb]
)

os.makedirs("models", exist_ok=True)
model.save("models/sac_franka_fig8_v6")
model.save_replay_buffer("models/sac_franka_fig8_v5_buffer")
print("Saved: models/sac_franka_fig8_v5 + buffer")

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import os
from collections import deque


class FrankaCircleEnv(gym.Env):
    """
    Franka Panda tracks a fixed figure-8 trajectory.
    End-effector moves to centre before tracking begins.

    Observation (34,):
        ee_pos(3), target_pos(3), ee_quat(4), target_quat(4),
        qpos(7), qvel(7), phase(2), traj_centre(3), traj_radius(1)

    Action (7,): delta joint positions
    """

    def __init__(self,
                 obs_noise_std = 0.005,
                 action_delay  = 1,
                 trajectory    = "figure8",
                 fixed_centre  = None,
                 fixed_radius  = None):
        super().__init__()

        self.obs_noise_std   = obs_noise_std
        self.action_delay    = action_delay
        self.trajectory_type = trajectory
        self.fixed_centre    = np.array(fixed_centre) if fixed_centre \
                               else np.array([0.5, 0.0, 0.5])
        self.fixed_radius    = fixed_radius if fixed_radius else 0.4

        # MuJoCo 
        model_path = os.path.join(
            "mujoco_menagerie-main",
            "franka_emika_panda",
            "scene.xml"
        )
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data  = mujoco.MjData(self.model)

        self.hand_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "hand"
        )

        # Trajectory 
        self.traj_centre  = self.fixed_centre.copy()
        self.traj_radius  = self.fixed_radius
        self.traj_speed   = 1.0
        self.traj_time    = 0.0
        self.tracking_started = False

        # Threshold to start tracking
        self.start_threshold = 0.05

        # Episode 
        self.max_steps  = 1000
        self.step_count = 0
        self.sim_dt     = self.model.opt.timestep * 5

        # Spaces
        self.action_space = spaces.Box(
            low=-0.05, high=0.05, shape=(7,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(34,), dtype=np.float32
        )

        # Control state
        self.joint_targets = np.zeros(7)
        self.prev_action   = np.zeros(7)
        self.ctrl_low  = self.model.actuator_ctrlrange[:7, 0]
        self.ctrl_high = self.model.actuator_ctrlrange[:7, 1]
        self.action_buffer = deque(maxlen=action_delay + 1)


    def _raw_target(self, phase):
        if self.trajectory_type == "figure8":
            return np.array([
                self.traj_centre[0],
                self.traj_centre[1] + self.traj_radius * np.sin(phase),
                self.traj_centre[2] + self.traj_radius * np.sin(2 * phase) / 2
            ])
        return np.array([
            self.traj_centre[0],
            self.traj_centre[1] + self.traj_radius * np.cos(phase),
            self.traj_centre[2] + self.traj_radius * np.sin(phase)
        ])

    def get_target(self):
        """
        Returns centre during acquisition phase.
        Returns trajectory point once tracking has started.
        Figure-8 always starts from centre.
        """
        if not self.tracking_started:
            return self.traj_centre.copy()
        return self._raw_target(self.traj_speed * self.traj_time)

    def get_orientation_target(self):
        if not self.tracking_started:
            return np.array([1.0, 0.0, 0.0, 0.0])

        phase = self.traj_speed * self.traj_time
        eps   = 1e-4

        tangent = self._raw_target(phase + eps) - self._raw_target(phase - eps)
        tangent_norm = np.linalg.norm(tangent)

        if tangent_norm < 1e-6:
            return np.array([1.0, 0.0, 0.0, 0.0])

        tangent = tangent / tangent_norm
        up = np.array([0.0, 0.0, 1.0])

        if abs(np.dot(tangent, up)) > 0.99:
            up = np.array([1.0, 0.0, 0.0])

        normal = up - np.dot(up, tangent) * tangent
        normal_norm = np.linalg.norm(normal)

        if normal_norm < 1e-6:
            return np.array([1.0, 0.0, 0.0, 0.0])

        normal  = normal / normal_norm
        z_axis  = np.array([0.0, 0.0, 1.0])
        axis    = np.cross(z_axis, normal)
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-6:
            return np.array([1.0, 0.0, 0.0, 0.0]) \
                   if np.dot(z_axis, normal) > 0 \
                   else np.array([0.0, 1.0, 0.0, 0.0])

        axis  = axis / axis_norm
        angle = np.arccos(np.clip(np.dot(z_axis, normal), -1.0, 1.0))
        w     = np.cos(angle / 2)
        xyz   = axis * np.sin(angle / 2)
        return np.array([w, xyz[0], xyz[1], xyz[2]])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        home = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0])
        self.data.qpos[:7] = home
        self.joint_targets  = home.copy()
        self.data.qpos[7:]  = 0.04
        mujoco.mj_forward(self.model, self.data)

        self.traj_centre      = self.fixed_centre.copy()
        self.traj_radius      = self.fixed_radius
        self.traj_time        = 0.0
        self.tracking_started = False
        self.step_count       = 0
        self.prev_action      = np.zeros(7)

        self.action_buffer.clear()
        for _ in range(self.action_delay + 1):
            self.action_buffer.append(np.zeros(7))

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -0.05, 0.05)

        self.action_buffer.append(action.copy())
        delayed_action = self.action_buffer[0]

        self.joint_targets = np.clip(
            self.joint_targets + delayed_action,
            self.ctrl_low, self.ctrl_high
        )
        self.data.ctrl[:7] = self.joint_targets

        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        ee_pos  = self.data.xpos[self.hand_id].copy()
        ee_quat = self.data.xquat[self.hand_id].copy()

        # Checking if arm has reached centre to start trajectory
        if not self.tracking_started:
            if np.linalg.norm(ee_pos - self.traj_centre) < self.start_threshold:
                self.tracking_started = True
        else:
            self.traj_time += self.sim_dt

        target   = self.get_target()
        tgt_quat = self.get_orientation_target()

        dist              = np.linalg.norm(target - ee_pos)
        dot               = np.clip(np.dot(ee_quat, tgt_quat), -1.0, 1.0)
        orientation_error = 1.0 - abs(dot)
        smoothness        = np.linalg.norm(action - self.prev_action)
        velocity_penalty  = np.linalg.norm(self.data.qvel[:7])

        reward = (
            - dist
            - 0.3  * abs(target[0] - ee_pos[0])
            - 0.1  * smoothness
            - 0.05 * velocity_penalty
        )

        self.prev_action = action.copy()
        self.step_count += 1
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), reward, False, truncated, {
            "dist":              dist,
            "orientation_error": orientation_error,
            "ee_pos":            ee_pos,
            "target":            target,
            "tracking_started":  self.tracking_started
        }

    def _get_obs(self):
        ee       = self.data.xpos[self.hand_id].copy()
        ee_quat  = self.data.xquat[self.hand_id].copy()
        target   = self.get_target()
        tgt_quat = self.get_orientation_target()
        qpos     = self.data.qpos[:7].copy()
        qvel     = self.data.qvel[:7].copy()

        phase     = self.traj_speed * self.traj_time
        phase_enc = np.array([np.sin(phase), np.cos(phase)])

        traj_info = np.array([*self.traj_centre, self.traj_radius])

        obs = np.concatenate([
            ee, target, ee_quat, tgt_quat,
            qpos, qvel, phase_enc, traj_info
        ]).astype(np.float32)

        if self.obs_noise_std > 0:
            noise = self.np_random.normal(
                0, self.obs_noise_std, size=obs.shape
            )
            obs = (obs + noise).astype(np.float32)

        return obs

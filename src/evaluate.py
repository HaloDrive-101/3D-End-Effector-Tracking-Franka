import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import SAC
from franka_traj_env import FrankaCircleEnv
import matplotlib.pyplot as plt


def run_episode(env, model, render=False):
    obs, _ = env.reset()

    ee_positions     = []
    target_positions = []
    orientations     = []
    distances        = []
    actions_taken    = []
    rewards          = []

    mj_model = env.model
    mj_data  = env.data

    if render:
        viewer_ctx = mujoco.viewer.launch_passive(mj_model, mj_data)
    else:
        viewer_ctx = None

    for step in range(800):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)

        if render and viewer_ctx:
            viewer_ctx.sync()

        ee_positions.append(info["ee_pos"].copy())
        target_positions.append(info["target"].copy())
        orientations.append(info["orientation_error"])
        distances.append(info["dist"])
        actions_taken.append(action.copy())
        rewards.append(reward)

        if term or trunc:
            break

    if render and viewer_ctx:
        viewer_ctx.close()

    return {
        "ee":        np.array(ee_positions),
        "targets":   np.array(target_positions),
        "dists":     np.array(distances),
        "actions":   np.array(actions_taken),
        "rewards":   np.array(rewards),
        "orientations": np.array(orientations)
    }


def compute_metrics(data, label=""):
    dists   = data["dists"]
    actions = data["actions"]

    orients = data.get("orientations", [])
    
    # Tracking metrics
    # Separate startup (first 100 steps) from steady-state
    startup_error      = dists[:100].mean()
    steadystate_error  = dists[100:].mean()

    # Smoothness: mean norm of consecutive action differences
    action_deltas = np.diff(actions, axis=0)
    smoothness    = np.mean(np.linalg.norm(action_deltas, axis=1))

    if len(orients) > 0:
        orients = np.array(orients)
        print(f"Mean orientation error:       {np.mean(orients):.4f}")
        print(f"Steady-state orient error:    {np.mean(orients[100:]):.4f}")

    print(f"\n=== {label} ===")
    print(f"Mean tracking error:          {dists.mean()*100:.2f} cm")
    print(f"  Startup phase  (0-100):     {startup_error*100:.2f} cm")
    print(f"  Steady-state   (100-1000):  {steadystate_error*100:.2f} cm")
    print(f"Max tracking error (0-1000):  {dists.max()*100:.2f} cm")
    print(f"Std tracking error (100-1000) {dists[100:].std()*100:.2f} cm")
    print(f"Smoothness (mean |Δaction|):  {smoothness:.5f}")
    print(f"Total reward:                 {data['rewards'].sum():.2f}")

    return {
        "mean":        dists.mean(),
        "startup":     startup_error,
        "steadystate": steadystate_error,
        "max":         dists.max(),
        "std":         dists[100:].std(),
        "smoothness":  smoothness
    }


# Load model
print("Loading robust model...")

# Run 1: noise + delay ON (training conditions) — with viewer
env_noisy = FrankaCircleEnv(
    obs_noise_std = 0.005,
    action_delay  = 1,
    trajectory    = "figure8",
    fixed_centre  = [0.5, 0.0, 0.5],
    fixed_radius  = 0.4
)
model = SAC.load("models/sac_franka_fig8_v5", env=env_noisy)

print("\nRunning evaluation with noise + delay (viewer open)")
data_noisy = run_episode(env_noisy, model, render=True)
metrics_noisy = compute_metrics(data_noisy, "Model — Noise + Delay ON")

# Run 2: noise OFF, delay OFF
env_clean = FrankaCircleEnv(
    obs_noise_std = 0.0,
    action_delay  = 0,
    trajectory    = "figure8",
    fixed_centre  = [0.5, 0.0, 0.5],
    fixed_radius  = 0.4
)
model_clean = SAC.load("models/sac_franka_fig8_v5", env=env_clean)

print("\nRunning evaluation without noise/delay...")
data_clean = run_episode(env_clean, model_clean, render=False)
metrics_clean = compute_metrics(data_clean, "Model — No Noise/Delay")


# ── Plots ──
# Figure 1: Top + Bottom tracking error plots
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle("Franka Circle Tracking — SAC Robust Model", fontsize=14)

# Plot 1: Tracking error over time — noisy+delay
ax = axes[0]
ax.plot(data_noisy["dists"], color="crimson", linewidth=0.8,
        alpha=0.7, label="Error (noise+delay)")
ax.axhline(metrics_noisy["steadystate"], color="navy", linestyle="--",
           label=f"Steady-state mean = {metrics_noisy['steadystate']*100:.1f}cm")
ax.axvline(100, color="gray", linestyle=":", alpha=0.5,
           label="Startup / steady boundary")
ax.set_title("Tracking Error — Noise + Delay ON")
ax.set_xlabel("Step")
ax.set_ylabel("Distance to target (m)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 2: Tracking error over time — clean
ax = axes[1]
ax.plot(data_clean["dists"], color="steelblue", linewidth=0.8,
        alpha=0.7, label="Error (no noise/delay)")
ax.axhline(metrics_clean["steadystate"], color="navy", linestyle="--",
           label=f"Steady-state mean = {metrics_clean['steadystate']*100:.1f}cm")
ax.axvline(100, color="gray", linestyle=":", alpha=0.5)
ax.set_title("Tracking Error — No Noise/Delay (same model)")
ax.set_xlabel("Step")
ax.set_ylabel("Distance to target (m)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results_tracking_errors.png", dpi=150)
print("\nSaved: results_tracking_errors.png")
plt.show()


# Figure 2: 3D trajectory
fig = plt.figure(figsize=(10, 8))
ax3d = fig.add_subplot(111, projection="3d")

ax3d.plot(data_noisy["targets"][:,0],
          data_noisy["targets"][:,1],
          data_noisy["targets"][:,2],
          "r--", linewidth=2, label="Target", alpha=0.7)

ax3d.plot(data_noisy["ee"][:,0],
          data_noisy["ee"][:,1],
          data_noisy["ee"][:,2],
          "b-", linewidth=1.2, label="Actual", alpha=0.9)

ax3d.set_title("3D Trajectory — Noise + Delay")
ax3d.set_xlabel("X")
ax3d.set_ylabel("Y")
ax3d.set_zlabel("Z")
ax3d.legend(fontsize=8)

plt.tight_layout()
plt.savefig("results_trajectory_3d.png", dpi=150)
print("Saved: results_trajectory_3d.png")
plt.show()


# Figure 3: Summary bar chart
fig, ax = plt.subplots(figsize=(8, 6))

labels = ["Noisy\n(train cond.)", "Clean\n(same model)"]
means  = [metrics_noisy["steadystate"]*100,
          metrics_clean["steadystate"]*100]
stds   = [metrics_noisy["std"]*100,
          metrics_clean["std"]*100]
colors = ["crimson", "steelblue"]

bars = ax.bar(labels, means, yerr=stds, color=colors,
              alpha=0.7, capsize=6, width=0.4)

ax.set_title("Steady-State Tracking Error Comparison")
ax.set_ylabel("Mean error (cm)")
ax.grid(True, alpha=0.3, axis="y")

for bar, val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.1,
            f"{val:.1f}cm", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig("results_summary_bar.png", dpi=150)
print("Saved: results_summary_bar.png")
plt.show()

print("\n=== Summary ===")
print(f"Steady-state error with noise+delay: "
      f"{metrics_noisy['steadystate']*100:.2f} cm")
print(f"Steady-state error without noise:    "
      f"{metrics_clean['steadystate']*100:.2f} cm")
print(f"Smoothness degradation due to noise: "
      f"{metrics_noisy['smoothness']:.5f} vs "
      f"{metrics_clean['smoothness']:.5f}")

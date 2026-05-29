import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
import os

# Loading workspace hull 
with open("franka_workspace_hull.pkl", "rb") as f:
    hull = pickle.load(f)

hull_points = hull.points[hull.vertices]

# Defining figure-eight trajectory
centre = np.array([0.5, 0.0, 0.5])
radius = 0.4
t      = np.linspace(0, 2 * np.pi, 500)

traj_y = centre[1] + radius * np.sin(t)
traj_z = centre[2] + radius * np.sin(2 * t) / 2
traj_x = np.full_like(t, centre[0])

# Plot region
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Figure-Eight Trajectory vs Franka Workspace", fontsize=13)

# --- XY plane (top view) ---
ax = axes[0]
ax.scatter(hull_points[:, 0], hull_points[:, 1],
           s=2, alpha=0.3, color="steelblue", label="Workspace boundary")
ax.plot(traj_x, traj_y, color="crimson", linewidth=2, label="Figure-eight")
ax.scatter(*centre[:2], color="black", s=60, zorder=5, label="Centre")
ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
ax.set_title("Top View (XY plane)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_aspect("equal")

# --- XZ plane (side view) ---
ax = axes[1]
ax.scatter(hull_points[:, 0], hull_points[:, 2],
           s=2, alpha=0.3, color="steelblue")
ax.plot(traj_x, traj_z, color="crimson", linewidth=2)
ax.scatter(centre[0], centre[2], color="black", s=60, zorder=5)
ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
ax.set_title("Side View (XZ plane)")
ax.grid(True, alpha=0.3); ax.set_aspect("equal")

# --- YZ plane (front view) ---
ax = axes[2]
ax.scatter(hull_points[:, 1], hull_points[:, 2],
           s=2, alpha=0.3, color="steelblue")
ax.plot(traj_y, traj_z, color="crimson", linewidth=2)
ax.scatter(centre[1], centre[2], color="black", s=60, zorder=5)
ax.set_xlabel("Y (m)"); ax.set_ylabel("Z (m)")
ax.set_title("Front View (YZ plane)")
ax.grid(True, alpha=0.3); ax.set_aspect("equal")

plt.tight_layout()
plt.savefig("trajectory_vs_workspace.png", dpi=150)
print("Saved: trajectory_vs_workspace.png")
plt.show()

# Printing coverage of workspace the trajectory uses 
ws_y_range = hull_points[:, 1].max() - hull_points[:, 1].min()
ws_z_range = hull_points[:, 2].max() - hull_points[:, 2].min()
traj_y_range = traj_y.max() - traj_y.min()
traj_z_range = traj_z.max() - traj_z.min()

print(f"\nWorkspace Y range: {ws_y_range:.3f}m")
print(f"Trajectory Y span: {traj_y_range:.3f}m  ({traj_y_range/ws_y_range*100:.1f}% of workspace)")
print(f"\nWorkspace Z range: {ws_z_range:.3f}m")
print(f"Trajectory Z span: {traj_z_range:.3f}m  ({traj_z_range/ws_z_range*100:.1f}% of workspace)")

workspace_delaunay = Delaunay(hull_points)

inside = workspace_delaunay.find_simplex(
    np.column_stack([traj_x, traj_y, traj_z])
) >= 0

print(f"\nTrajectory points inside workspace: {inside.sum()}/{len(inside)}")
print(f"All reachable: {inside.all()}")

if not inside.all():
    outside_idx = np.where(~inside)[0]
    print(f"First unreachable point: y={traj_y[outside_idx[0]]:.3f}, "
          f"z={traj_z[outside_idx[0]]:.3f}")

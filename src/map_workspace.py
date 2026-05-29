import mujoco
import numpy as np
import os
from scipy.spatial import ConvexHull
import pickle
import matplotlib.pyplot as plt

model_path = os.path.join(
    "mujoco_menagerie-main",
    "franka_emika_panda",
    "scene.xml"
)
model = mujoco.MjModel.from_xml_path(model_path)
data  = mujoco.MjData(model)

hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")

# Joint limits for arm joints 0-6
ctrl_low  = model.actuator_ctrlrange[:7, 0]
ctrl_high = model.actuator_ctrlrange[:7, 1]

print("Sampling workspace...")
workspace_points = []
n_samples = 1_000_000

for i in range(n_samples):
    if i % 20000 == 0:
        print(f"  {i}/{n_samples}")

    # Random joint configuration within limits
    q = np.random.uniform(ctrl_low, ctrl_high)

    # Set joints and run forward kinematics
    data.qpos[:7] = q
    mujoco.mj_forward(model, data)

    ee_pos = data.xpos[hand_id].copy()
    workspace_points.append(ee_pos)

workspace_points = np.array(workspace_points)
print(f"Sampled {len(workspace_points)} points")

# Fit convex hull
print("Fitting convex hull...")
hull = ConvexHull(workspace_points)
print(f"Hull has {len(hull.vertices)} vertices")

# Save hull for use in environment
with open("franka_workspace_hull.pkl", "wb") as f:
    pickle.dump(hull, f)

print("Saved: franka_workspace_hull.pkl")

# Mark workspace bounds
print(f"\nWorkspace bounds:")
print(f"  X: [{workspace_points[:,0].min():.3f}, {workspace_points[:,0].max():.3f}]")
print(f"  Y: [{workspace_points[:,1].min():.3f}, {workspace_points[:,1].max():.3f}]")
print(f"  Z: [{workspace_points[:,2].min():.3f}, {workspace_points[:,2].max():.3f}]")

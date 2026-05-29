A)Directory Summary-

  src/              — environment, training, evaluation code
  
  models/           — pre-trained SAC model + replay buffer
  
  results/plots/    — tracking error plots, 3D trajectory, workspace
  
  results/videos/   — demo video
  
  mujoco_menagerie-main/  — Franka Panda MuJoCo model

B)Instructions-

  NOTE: Before running any code make sure to change filename in code to prevent overwrite of submitted files/videos/plots.

  1. Clone the repository:

  --git clone https://github.com/HaloDrive-101/3D-End-Effector-Tracking-Franka.git

  --cd rl_traj_proj

  2. Install dependencies:

  --pip install -r requirements.txt

  3. Run evaluation (uses pre-trained model):

  --python src/evaluate.py

     This opens the MuJoCo viewer showing the arm tracking the
     figure-eight shape, then saves plots to `results/plots/`.

  5. If you are training From scratch:

     For visualizing the trajectory in the Franka workspace:

  --python src/map_workspace.py   (also generates franka_workspace_hull.pkl)

     Wait few minutes for workspace sampling to complete.

  --python src/plot_workspace.py

     Edit the file name for the new model and buffer in src/train_traj_sac.py

  --python src/train_traj_sac.py

     Training runs for 1M steps (~6-7 hours on my CPU). Model saved to `models/YOUR_FILENAME`.

C)Technical Notes-

   ## Overview

   A Franka Panda 7-DOF arm learns to track a figure-eight end-effector trajectory in 3D Cartesian space using Soft Actor-Critic (SAC). This system handles sensor noise, control delay, and maintains a stable motion throughout tracking.

   ## State (Observation) 

   Observation consists of 8 states with a total of 34 dimensions-

   1| EE position                | 3 | Where the end effector is 
   2| Target position            | 3 | Where the end effector should be 
   3| EE quaternion              | 4 | Current wrist orientation 
   4| Target quaternion          | 4 | Desired perpendicular orientation 
   5| Joint positions            | 7 | -
   6| Joint velocities           | 7 | - 
   7| Phase [sin φ, cos φ]       | 2 | Trajectory progress
   8| Trajectory centre + radius | 4 | Trajectory shape parameters 

   -Phase is encoded as sin/cos rather than a raw angle so as to avoid discontinuities at 2π and to let the agent anticipate target motion direction.

   -The hand quaternion and target quaternion are included so the agent can reason about wrist configuration, even though orientation is not  rewarded. The target orientation state in this case is an example of an application that can be used, where the end effector is normal to the trajectory at each point, like its holding a pen perpendicular to the curve being drawn.

   ## Action Space

   -The 7-dimensional joint positions (for 7DOF) here, are bounded to ±0.05 radians per step. The actions are controlled in increments,  rather than absolute positions. This naturally limits maximum joint speed, and provides built-in motion smoothness.

   ## Reward Function

   The reward function contains 4 terms-

   1) Tracking distance term - This is the primary signal, which is the euclidean distance to target.
   2) X-axis penalty         - The figure-eight lies in the YZ plane with fixed X = 0.5m. Without this term, the agent occasionally drifts 
                              in depth since X error contributes less gradient than fast-moving Y/Z during active trajectory segments.
   3) Smoothness term        - This term penalises large consecutive action changes, directly discouraging jitter.
   4) Velocity term          - This term penalises fast joint motion, encouraging deliberate rather than aggressive control, which results in
                              stable behaviour.

   ## Trajectory Representation

   -A figure-8 shape is selected for end effector trajectory.

   -The trajectory is stationary at the centre point of the figure-8 until the end-effector arrives within 5cm of it. This removes the startup transient from the tracking evaluation and produces cleaner and more physically meaningful episodes.

   -The complete figure-eight was verified to lie within the Franka reachable workspace using a convex hull fitted to 1M forward kinematics samples (random joint configurations). The visualization of the trajectory shape within the workspace can be viewed in one of the plots.

   ## Uncertainty Sources Implemented

   1) Observation noise: Gaussian signal was added to all 34 observation dimensions every step, which models imperfect sensor readings in 
                        real life.
   2) Action delay:      The action commands are applied 1 step late via a FIFO buffer. The agent doesnt observe immediate effects of its 
                        actions.

   ## Evaluation of Tracking Performance

   The tracking error is measured as Euclidean distance between end-effector and target, which is split into two phases:

   1) Startup (steps 0–100): The arm is moving from home to trajectory position (end effector heading towards the centre of the figure-8). 
      Higher error expected and excluded from active tracking error reporting.
   2) Trajectory Tracking (steps > 100): Active tracing of the figure-8 shape 

   The following evaluation of performance metrics is averaged over 3 evaluation episodes-

   | Condition           | Steady-state error during tracking phase | Smoothness |

   | With noise + delay  |                  1.31 cm                 |   0.061    |
   | Without noise/delay |                  1.09 cm                 |   0.028    |

   Smoothness is measured as the mean norm of consecutive action differences (||a_t - a_{t-1}||) across all 7 joints per episode.

   Smoothness doubles under sensor noise because the agent makes micro-corrections in response to noisy state readings. This is an expected tradeoff showing that the agent is reactive to its sensors. For future works, implmenting a Kalman filter on observations would reduce this. Further tuning of reward design can also accomplish the same.

   The trajectory radius used is 0.4m. Steady-state error of 1.09cm without uncertainty represents 2.7% relative error. The 0.22cm increase 
   under noise and delay demonstrates that the policy has learned effective compensation for both uncertainty sources.

   | Episode | Steady-state (noisy+delay) | Steady-state (clean) |

   |    1    |          1.31 cm           |        1.09 cm       |
   |    2    |          1.31 cm           |        1.09 cm       |
   |    3    |          1.32 cm           |        1.09 cm       |

   The noisy column results show < 0.01cm variance across the 3 runs. This confirms stable converged behaviour rather than lucky evaluation episodes.

   | Episode | Std tracking error (noisy+delay) | Std tracking error (clean) |

   |    1    |             0.52 cm              |          0.48 cm           |
   |    2    |             0.58 cm              |          0.48 cm           |
   |    3    |             0.58 cm              |          0.48 cm           |

   Around these steady-state mean errors, the standard tracking error remains low, varying between 0.52–0.58 cm for the noisy case and staying constant at 0.48 cm for the clean case, demonstrating consistent and repeatable performance despite disturbances.

D)Code Example Output [Running evaluate.py code for analysis] - 

   (rl_env) C:\Users\abhig\rl_project>python evaluate.py
   Loading robust model...
   Wrapping the env with a `Monitor` wrapper
   Wrapping the env in a DummyVecEnv.

   Running evaluation WITH noise + delay (viewer open)...
   Mean orientation error:       0.5737
   Steady-state orient error:    0.5402

   === Robust Model — Noise + Delay ON ===
   Mean tracking error:          1.59 cm
   Startup phase  (0-100):     3.82 cm
   Steady-state   (100-1000):  1.28 cm
   Max tracking error (0-1000):  19.01 cm
   Std tracking error (100-1000) 0.58 cm
   Smoothness (mean |Δaction|):  0.06011
   Total reward:                 -72.25
   Wrapping the env with a `Monitor` wrapper
   Wrapping the env in a DummyVecEnv.

   Running evaluation WITHOUT noise/delay...
   Mean orientation error:       0.5730
   Steady-state orient error:    0.5392

   === Robust Model — No Noise/Delay ===
   Mean tracking error:          1.43 cm
   Startup phase  (0-100):     3.82 cm
   Steady-state   (100-1000):  1.09 cm
   Max tracking error (0-1000):  18.91 cm
   Std tracking error (100-1000) 0.48 cm
   Smoothness (mean |Δaction|):  0.02834
   Total reward:                 -52.32

   Saved: results_tracking_errors.png
   Saved: results_trajectory_3d.png
   Saved: results_summary_bar.png

   === Summary ===
   Steady-state error with noise+delay: 1.28 cm
   Steady-state error without noise:    1.09 cm
   Smoothness degradation due to noise: 0.06011 vs 0.02834

import os, sys
sys.path.insert(0, r"scene_magnet")
from envs.arm_env import ArmEnv
e = ArmEnv()
print("scene_magnet nbody", e.model.nbody, "nu", e.model.nu, "nq", e.model.nq)

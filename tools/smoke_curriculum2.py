import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.arm_env import ArmEnv, CurriculumCallback, WaitAwareMonitor

def run(threshold, steps, tag):
    env=DummyVecEnv([lambda: WaitAwareMonitor(ArmEnv()) for _ in range(8)])
    m=PPO.load("scene_magnet/saves/phase1.zip", env=env, device="cpu", learning_rate=1e-4)
    cb=CurriculumCallback(min_steps=6000, success_threshold=threshold,
                          probe_episodes=8, probe_every=6000, verbose=1)
    print("=== %s (阈值 %.2f) ===" % (tag, threshold))
    m.learn(total_timesteps=steps, callback=[cb])
    print("最终档位 idx =", cb.idx)

run(0.40, 30000, "低阈值")
run(0.99, 30000, "高阈值")

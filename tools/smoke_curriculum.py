import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from envs.arm_env import ArmEnv, CurriculumCallback, WaitAwareMonitor

def run(threshold, steps, tag):
    env=DummyVecEnv([lambda: WaitAwareMonitor(ArmEnv()) for _ in range(8)])
    m=PPO.load("scene_magnet/saves/phase1.zip", env=env, device="cpu", learning_rate=1e-4)
    cb=CurriculumCallback(min_steps=8000, success_threshold=threshold, verbose=1)
    ck=CheckpointCallback(save_freq=max(1,5000//8), save_path="scene_magnet/saves/_ckpt_test",
                          name_prefix=tag, save_vecnormalize=False)
    print("=== %s (阈值 %.2f) ===" % (tag, threshold))
    m.learn(total_timesteps=steps, callback=[cb, ck])
    print("最终档位 idx =", cb.idx, "| 本档局数", cb.ep_total, "| 成功", cb.ep_ok)

run(0.30, 40000, "loose")
run(0.99, 40000, "strict")
print("检查点文件:", sorted(os.listdir("scene_magnet/saves/_ckpt_test")))

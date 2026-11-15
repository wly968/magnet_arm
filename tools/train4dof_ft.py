import os, sys, time
os.environ["OMP_NUM_THREADS"]="1"
import numpy as np, torch
torch.set_num_threads(1)
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from envs.arm_env import ArmEnv, CurriculumCallback

# 窄课程：从旧模型已有水平起步，逐步压到终点
CurriculumCallback.SCHEDULE = [
    (0,         0.040, 0.950, 1),
    (600_000,   0.035, 0.955, 1),
    (1_200_000, 0.030, 0.960, 2),
]
N=24; STEPS=3_000_000
env = DummyVecEnv([lambda: Monitor(ArmEnv()) for _ in range(N)])
eval_env = DummyVecEnv([lambda: Monitor(ArmEnv())])
eval_env.env_method("_set_dist", 0.03); eval_env.env_method("_set_angle1", 0.96); eval_env.env_method("_set_still_count", 2)

model = PPO.load(os.path.join(SCENE,"ppo_arm_reach.zip"), env=env, device="cpu", learning_rate=1e-4)
print("从 ppo_arm_reach 微调，窄课程")
cb = EvalCallback(eval_env, best_model_save_path=os.path.join(SCENE,"saves","best_4dof_ft"),
                  log_path=os.path.join(SCENE,"saves","eval_logs_4dof_ft"),
                  eval_freq=max(2000, 50_000 // N), n_eval_episodes=20,
                  deterministic=True, verbose=1)
t0=time.time()
model.learn(total_timesteps=STEPS, callback=[CurriculumCallback(verbose=1), cb])
model.save(os.path.join(SCENE,"saves","arm4dof_ft.zip"))
print("FT DONE in %.0f s" % (time.time()-t0))

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

N=24; STEPS=10_000_000
env = DummyVecEnv([lambda: Monitor(ArmEnv()) for _ in range(N)])

# eval 环境固定为终点阈值，best_model 才有意义
eval_env = DummyVecEnv([lambda: Monitor(ArmEnv())])
eval_env.env_method("_set_dist", 0.03)
eval_env.env_method("_set_angle1", 0.96)
eval_env.env_method("_set_still_count", 2)

dst = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=512, batch_size=1024,
          n_epochs=5, gamma=0.99, gae_lambda=0.95, ent_coef=0.005, device="cpu",
          policy_kwargs=dict(net_arch=[128,128]), seed=0, target_kl=0.02)

# 4 维直接热启动
src = PPO.load(os.path.join(SCENE,"ppo_arm_reach.zip"), device="cpu")
sd_src=src.policy.state_dict(); sd_dst=dst.policy.state_dict()
n=0
for k,v in sd_src.items():
    if k in sd_dst and sd_dst[k].shape==v.shape:
        sd_dst[k]=v.clone(); n+=1
dst.policy.load_state_dict(sd_dst)
print("热启动拷贝张量数:", n)

cb = EvalCallback(eval_env, best_model_save_path=os.path.join(SCENE,"saves","best_4dof"),
                  log_path=os.path.join(SCENE,"saves","eval_logs_4dof"),
                  eval_freq=max(2000, 100_000 // N), n_eval_episodes=20,
                  deterministic=True, verbose=1)
t0=time.time()
dst.learn(total_timesteps=STEPS, callback=[CurriculumCallback(verbose=1), cb])
dst.save(os.path.join(SCENE,"saves","arm4dof_10m.zip"))
print("4DOF 10M DONE in %.0f s" % (time.time()-t0))

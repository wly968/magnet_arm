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
eval_env = DummyVecEnv([lambda: Monitor(ArmEnv())])

dst = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=512, batch_size=1024,
          n_epochs=5, gamma=0.99, gae_lambda=0.95, ent_coef=0.005, device="cpu",
          policy_kwargs=dict(net_arch=[128,128]), seed=0, target_kl=0.02)

src = PPO.load(os.path.join(SCENE,"ppo_arm_reach.zip"), device="cpu")
sd_src=src.policy.state_dict(); sd_dst=dst.policy.state_dict()
copied=padded=0
for k,v in sd_src.items():
    if k not in sd_dst: continue
    tgt=sd_dst[k]
    if tgt.shape==v.shape: sd_dst[k]=v.clone(); copied+=1
    else:
        new=tgt.clone()
        if v.dim()==2 and v.shape[1]==tgt.shape[1]:
            new[:v.shape[0],:]=v; new[v.shape[0]:,:]=0.0
        elif v.dim()==1 and v.shape[0]<tgt.shape[0]:
            new[:v.shape[0]]=v
            new[v.shape[0]:]=-1.0 if "log_std" in k else 0.0
        sd_dst[k]=new; padded+=1
dst.policy.load_state_dict(sd_dst)
print(f"热启动: 拷贝 {copied}, 扩维 {padded}")

cb = EvalCallback(eval_env, best_model_save_path=os.path.join(SCENE,"saves","best_5dof"),
                  log_path=os.path.join(SCENE,"saves","eval_logs_5dof"),
                  eval_freq=max(2000, 100_000 // N), n_eval_episodes=10,
                  deterministic=True, verbose=1)
t0=time.time()
dst.learn(total_timesteps=STEPS, callback=[CurriculumCallback(verbose=1), cb])
dst.save(os.path.join(SCENE,"saves","arm5dof_10m.zip"))
print("10M DONE in %.0f s" % (time.time()-t0))

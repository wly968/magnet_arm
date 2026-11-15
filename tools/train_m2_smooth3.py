import os, sys, time, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"
torch.set_num_threads(1)
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from envs.arm_env import ArmEnv, CurriculumCallback
from envs.phase34 import Phase34Env
m1=PPO.load(os.path.join(SCENE,"ppo_arm_reach.zip"), device="cpu")
def make(): return Monitor(Phase34Env(ArmEnv(terminate_on_phase2=False, max_phase=6), m1))
env=DummyVecEnv([make for _ in range(24)])
m2=PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4, n_steps=512, batch_size=1024,
       n_epochs=5, gamma=0.99, gae_lambda=0.95, ent_coef=0.01, device="cpu",
       policy_kwargs=dict(net_arch=[128,128]), seed=1, target_kl=0.02)
CurriculumCallback.SCHEDULE=[(0,0.065,0.930,0),(500_000,0.055,0.940,0),(1_500_000,0.045,0.950,1)]
t0=time.time(); m2.learn(total_timesteps=3_000_000, callback=CurriculumCallback()); print("model2 %.0fs"%(time.time()-t0))
m2.save(os.path.join(SCENE,"saves","phase345_smooth3.zip"))
e=ArmEnv(terminate_on_phase2=False, max_phase=6); e.max_steps=300
e._set_dist(0.05); e._set_angle1(0.945); e._set_still_count(0)
ph={}
for ep in range(40):
    e.reset(); e.phase=3
    for t in range(e.max_steps):
        o=e._get_obs(); a,_=m2.predict(o,deterministic=True)
        o,r,term,tr,info=e.step(a)
        if term or tr: break
    ph[e.phase]=ph.get(e.phase,0)+1
print("训练后自测 从phase3: %s" % ph)

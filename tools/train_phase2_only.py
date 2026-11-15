import os, sys, time, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"
torch.set_num_threads(1)
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from envs.arm_env import ArmEnv, CurriculumCallback
from envs.phase2 import Phase2Env

m1 = PPO.load(os.path.join(SCENE,"saves","phase1.zip"), device="cpu")
def make():
    return Monitor(Phase2Env(ArmEnv(), m1))

env=DummyVecEnv([make for _ in range(24)])
m=PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4, n_steps=512, batch_size=1024,
      n_epochs=5, gamma=0.99, gae_lambda=0.95, ent_coef=0.01, device="cpu",
      policy_kwargs=dict(net_arch=[128,128]), seed=0, target_kl=0.02)
CurriculumCallback.SCHEDULE=[(0,0.060,0.930,0),(400_000,0.045,0.950,1),(900_000,0.035,0.960,1)]
t0=time.time(); m.learn(total_timesteps=2_000_000, callback=CurriculumCallback()); print("2M 训练 %.0fs"%(time.time()-t0))
m.save(os.path.join(SCENE,"saves","phase2_only.zip"))

ev=ArmEnv(); ev.max_steps=250
for d_,a_,s_ in ((0.035,0.96,1),(0.040,0.96,1)):
    ev._set_dist(d_); ev._set_angle1(a_); ev._set_still_count(s_)
    ok=0; steps=[]
    for ep in range(150):
        o,_=ev.reset()
        for t in range(ev.max_steps):
            act,_=m.predict(o,deterministic=True)
            o,r,term,tr,info=ev.step(act)
            if term or tr: break
        ok+=int(ev.phase>=3); steps.append(t+1)
    print("  %.3f/%.2f/%d -> %3d/150 = %5.1f%%"%(d_,a_,s_,ok,100*ok/150))

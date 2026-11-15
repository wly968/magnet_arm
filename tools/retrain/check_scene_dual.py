import os, sys
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv()
m1=PPO.load(os.path.join(SCENE,"saves","phase2.zip"), env=env)
m2=PPO.load(os.path.join(SCENE,"saves","model2_p3.zip"), env=env)
N=30; succ=0; ends={}
for ep in range(N):
    obs,_=env.reset(); term=False
    for t in range(env.max_steps):
        policy = m1 if env.phase < 3 else m2
        a,_=policy.predict(obs,deterministic=True)
        obs,r,term,trunc,info=env.step(a)
        if term or trunc: break
    ends[env.phase]=ends.get(env.phase,0)+1; succ+=int(env.phase==4)
print("dual scene check:", f"{succ}/{N}", ends)

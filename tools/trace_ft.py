import os, sys, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv(); model=PPO.load(os.path.join(ROOT,"tools","retrain","phase3_ft"), env=env)
obs,_=env.reset(seed=0)
a=env.box_qpos_adr; env.data.qpos[a:a+3]=[0,0.6,0.375]; env.data.qpos[a+3:a+7]=[1,0,0,0]
env.data.qvel[env.box_qvel_adr:env.box_qvel_adr+6]=0; env.box_pos=np.array([0,0.6,0.375])
mujoco.mj_forward(env.model,env.data); obs=env._get_obs()
for t in range(150):
    act,_=model.predict(obs,deterministic=True)
    obs,r,term,trunc,info=env.step(act)
    if env.phase==3 and t%10==0:
        d=env.data; ee=d.site_xpos[env.ee_id]; q=d.qpos[env.finger_qpos_adr]
        print(f"t={t} ee=({ee[0]:+.2f},{ee[1]:+.2f},{ee[2]:.3f}) q={q:.4f} act5={act[5]:+.2f} r={r:.1f}")
    if term or trunc: break
print("end phase",env.phase,"term",term,"steps",t+1)

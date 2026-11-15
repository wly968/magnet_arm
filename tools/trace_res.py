import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv(); m=PPO.load(os.path.join(SCENE,"saves","phase2.zip"), env=env)
obs,_=env.reset(seed=0)
a=env.box_qpos_adr; env.data.qpos[a:a+3]=[0,0.6,0.375]; env.data.qpos[a+3:a+7]=[1,0,0,0]
env.data.qvel[env.box_qvel_adr:env.box_qvel_adr+6]=0; env.box_pos=np.array([0,0.6,0.375])
import mujoco; mujoco.mj_forward(env.model,env.data); obs=env._get_obs()
for t in range(80):
    act,_=m.predict(obs,deterministic=True)
    obs,r,term,trunc,info=env.step(act)
    if env.phase==3 and t%10==0:
        d=env.data; ee=d.site_xpos[env.ee_id]; b=env.box_pos; tgt=b+np.array([0,0,0.02])
        print(f"t={t} ee={np.round(ee,3)} box={np.round(b,3)} dz={abs(ee[2]-tgt[2]):.3f} dxy={np.linalg.norm(ee[:2]-tgt[:2]):.3f} q={d.qpos[env.finger_qpos_adr]:.4f} r={r:.1f}")
    if term or trunc: break

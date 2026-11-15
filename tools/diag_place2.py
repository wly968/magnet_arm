import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m2=PPO.load(os.path.join(SCENE,"saves","phase345.zip"), device="cpu")
for still in (0,1):
    e=ArmEnv(terminate_on_phase2=False, max_phase=6); e.max_steps=300
    e._set_dist(0.05); e._set_angle1(0.945); e._set_still_count(still)
    ph={}; vs=[]
    for ep in range(60):
        e.reset(); e.phase=3
        for t in range(e.max_steps):
            o=e._get_obs(); a,_=m2.predict(o,deterministic=True)
            o,r,term,tr,info=e.step(a)
            if e.phase==5:
                ee=e.data.site_xpos[e.ee_id]
                if np.linalg.norm(ee-e._target())<0.05:
                    vs.append(float(np.linalg.norm(e.data.qvel[e.arm_dof_adr])))
            if term or tr: break
        ph[e.phase]=ph.get(e.phase,0)+1
    v=np.array(vs) if vs else np.array([np.nan])
    print("still=%d: %s, 到位时 qvel 中位 %.3f (n=%d)" % (still, ph, np.nanmedian(v), len(vs)))

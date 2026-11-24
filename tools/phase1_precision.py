import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
for name in ('ppo_arm_reach.zip','saves/phase1.zip'):
    m=PPO.load('scene_magnet/'+name, device='cpu')
    e=ArmEnv()
    print("=== %s ===" % name)
    for d,a in ((0.10,0.90),(0.07,0.94),(0.05,0.96),(0.04,0.97),(0.03,0.98)):
        e._set_dist(d); e._set_angle1(a); e._set_still_count(0); e._set_rel(0.05)
        ok=0; mins=[]
        for ep in range(20):
            obs,_=e.reset(); md=9
            for t in range(e.max_steps+e.max_wait_steps+10):
                act,_=m.predict(obs,deterministic=True)
                obs,r,term,tr,info=e.step(act)
                if e.phase<2:
                    ee=e.data.site_xpos[e.ee_id]; tgt=e.box_pos+e.pick_offset
                    md=min(md,float(np.linalg.norm(ee-tgt)))
                if e.phase>=2 or term or tr: break
            ok+=int(e.phase>=2); mins.append(md)
        print("  %.2f/%.2f -> phase1 %2d/20, 最小dist 中位 %.3f" % (d,a,ok,np.median(mins)))

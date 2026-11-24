import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
print("步数", getattr(m,'num_timesteps',0), "| obs", m.observation_space.shape)
for (d,a,s,rv) in ((0.050,0.94,0,0.20),(0.042,0.96,1,0.10),(0.034,0.98,2,0.05),(0.030,0.99,2,0.05)):
    e=ArmEnv(); e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e._set_rel(rv)
    ok=0; rels=[]; pos=0; tot=0
    for ep in range(30):
        obs,_=e.reset()
        for t in range(e.max_steps+e.max_wait_steps+10):
            act,_=m.predict(obs,deterministic=True)
            obs,r,term,tr,info=e.step(act)
            if e.phase==2:
                tot+=1; rels.append(float(np.linalg.norm(e.ee_val-e.box_vel)))
                ee=e.data.site_xpos[e.ee_id]; tgt=e.box_pos+e.grasp_offset
                dd=float(np.linalg.norm(ee-tgt))
                j3=e.data.xanchor[e.arm_joints[3]]; f=(ee-j3)/np.linalg.norm(ee-j3)
                if dd<d and float(np.dot(f,[0,0,-1]))>a: pos+=1
            if e.phase>=3 or term or tr: break
        ok+=int(e.phase>=3)
    print("  %.3f/%.2f/still%d/rel%.2f -> 成功 %2d/30 (%3.0f%%) | 位置到位 %4.1f%% | rel 中位 %.3f"
          % (d,a,s,rv,ok,100*ok/30,100*pos/max(1,tot),np.median(rels) if rels else float('nan')))

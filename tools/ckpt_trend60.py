import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
e=ArmEnv(); e._set_dist(0.032); e._set_angle1(0.985); e._set_still_count(1); e._set_rel(0.05)
for p,tag in (("scene_magnet/saves/ckpt/ckpt_499992_steps.zip","50万"),
              ("scene_magnet/ppo_arm_reach.zip","最终")):
    m=PPO.load(p, device="cpu")
    ok=0; angs=[]
    for ep in range(60):
        obs,_=e.reset()
        for t in range(e.max_steps+e.max_wait_steps+10):
            act,_=m.predict(obs,deterministic=True)
            obs,r,term,tr,info=e.step(act)
            if e.phase==2:
                ee=e.data.site_xpos[e.ee_id]; j3=e.data.xanchor[e.arm_joints[3]]
                f=(ee-j3)/np.linalg.norm(ee-j3); angs.append(float(np.dot(f,[0,0,-1])))
            if e.phase>=3 or term or tr: break
        ok+=int(e.phase>=3)
    print("%-6s 60 局: %2d/60 = %3.0f%% | phase2 内 angle 中位 %.3f" % (tag, ok, 100*ok/60, np.median(angs)))

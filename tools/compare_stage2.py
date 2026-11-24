import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
for name in ("scene_magnet/saves/phase1.zip","scene_magnet/saves/ckpt/ckpt_2999952_steps.zip"):
    m=PPO.load(name, device='cpu')
    e=ArmEnv(); e._set_dist(0.045); e._set_angle1(0.95); e._set_still_count(0); e._set_rel(0.10)
    ok=0; rels=[]
    for ep in range(30):
        obs,_=e.reset()
        for t in range(e.max_steps+e.max_wait_steps+10):
            a,_=m.predict(obs,deterministic=True)
            obs,r,term,tr,info=e.step(a)
            if e.phase==2: rels.append(float(np.linalg.norm(e.ee_val-e.box_vel)))
            if e.phase>=3 or term or tr: break
        ok+=int(e.phase>=3)
    print("%-42s 成功 %2d/30 | rel 中位 %.3f" % (os.path.basename(name), ok, np.median(rels) if rels else float("nan")))

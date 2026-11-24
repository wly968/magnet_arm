import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
cands=["scene_magnet/saves/phase1.zip",
       "scene_magnet/saves/ckpt/ckpt_1499976_steps.zip",
       "scene_magnet/saves/ckpt/ckpt_2999952_steps.zip",
       "scene_magnet/saves/best_4dof/best_model.zip"]
for name in cands:
    if not os.path.exists(name): print(name,"不存在"); continue
    m=PPO.load(name, device='cpu')
    e=ArmEnv(); e._set_dist(0.045); e._set_angle1(0.95); e._set_still_count(0); e._set_rel(0.10)
    ok=0; rels=[]; pos=0; tot=0
    for ep in range(30):
        obs,_=e.reset()
        for t in range(e.max_steps+e.max_wait_steps+10):
            a,_=m.predict(obs,deterministic=True)
            obs,r,term,tr,info=e.step(a)
            if e.phase==2:
                tot+=1; rels.append(float(np.linalg.norm(e.ee_val-e.box_vel)))
            if e.phase>=3 or term or tr: break
        ok+=int(e.phase>=3)
    print("%-40s 步数 %8s | 成功 %2d/30 | rel 中位 %.3f | rel<0.1 占比 %4.1f%%"
          % (os.path.basename(name), getattr(m,'num_timesteps',0), ok,
             np.median(rels) if rels else float('nan'),
             100*np.mean(np.array(rels)<0.10) if rels else 0))

import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
for p in ("scene_magnet/saves/best_4dof/best_model.zip",):
    m=PPO.load(p, device='cpu')
    print(os.path.basename(p), "步数", getattr(m,'num_timesteps',0), "obs", m.observation_space.shape)
    e=ArmEnv()
    for (d,a,s,rv) in ((0.034,0.98,2,0.05),(0.030,0.99,2,0.05),(0.032,0.985,1,0.05)):
        e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e._set_rel(rv)
        ok=0
        for ep in range(20):
            obs,_=e.reset()
            for t in range(e.max_steps+e.max_wait_steps+10):
                act,_=m.predict(obs,deterministic=True)
                obs,r,term,tr,info=e.step(act)
                if e.phase>=3 or term or tr: break
            ok+=int(e.phase>=3)
        print("  %.3f/%.3f/still%d/rel%.2f -> %2d/20" % (d,a,s,rv,ok))

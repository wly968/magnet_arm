import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
for (d,a,s,tag) in ((0.030,0.990,1,"test.py 当前 TH_PHASE12"),
                    (0.032,0.980,1,"训练第6档"),
                    (0.034,0.980,2,"训练第5档")):
    e=ArmEnv(); e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e._set_rel(0.05)
    ok=0; p2=0
    for ep in range(40):
        obs,_=e.reset()
        for t in range(e.max_steps+e.max_wait_steps+10):
            act,_=m.predict(obs,deterministic=True)
            obs,r,term,tr,info=e.step(act)
            if e.phase>=3 or term or tr: break
        p2+=int(e.phase>=2); ok+=int(e.phase>=3)
    print("%-24s %.3f/%.3f/still%d -> 进phase2 %2d/40 (%3.0f%%) | 成功 %2d/40 (%3.0f%%)"
          % (tag,d,a,s,p2,100*p2/40,ok,100*ok/40))

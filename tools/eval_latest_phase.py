import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load("scene_magnet/ppo_arm_reach.zip", device="cpu")
print("步数", getattr(m,'num_timesteps',0), "| obs", m.observation_space.shape)
for (d,a,s,rv,tag) in ((0.050,0.94,0,0.20,"第1档"),
                       (0.034,0.98,2,0.05,"第5档"),
                       (0.032,0.985,1,0.05,"第6档")):
    e=ArmEnv(); e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e._set_rel(rv)
    p1=p2=ok=0
    for ep in range(30):
        obs,_=e.reset()
        for t in range(e.max_steps+e.max_wait_steps+10):
            act,_=m.predict(obs,deterministic=True)
            obs,r,term,tr,info=e.step(act)
            if e.phase>=3 or term or tr: break
        p1+=int(e.phase>=2); ok+=int(e.phase>=3)
    print("  %s %.3f/%.2f/still%d -> 进phase2 %2d/30 (%3.0f%%) | 成功 %2d/30 (%3.0f%%)"
          % (tag,d,a,s,p1,100*p1/30,ok,100*ok/30))

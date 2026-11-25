import os, sys, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv, CurriculumCallback
m=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
print("步数", getattr(m,'num_timesteps',0), "| obs", m.observation_space.shape)
SCH=CurriculumCallback.SCHEDULE
for i,(d,a,s,rv) in enumerate(SCH):
    e=ArmEnv(); e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e._set_rel(rv)
    ok=0; p2=0
    for ep in range(30):
        obs,_=e.reset()
        for t in range(1400):
            act,_=m.predict(obs,deterministic=True)
            obs,r,term,tr,info=e.step(act)
            if e.phase>=3 or term or tr: break
        p2+=int(e.phase>=2); ok+=int(e.phase>=3)
    print("  第%d档 %.3f/%.3f/still%d/rel%.2f -> 进phase2 %2d/30 | 抓取成功 %2d/30 = %3.0f%%"
          % (i+1,d,a,s,rv,p2,ok,100*ok/30))

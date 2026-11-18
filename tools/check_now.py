import sys, os, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load(r"scene_magnet/saves/phase1.zip", device="cpu")
for d,a,s in ((0.06,0.90,0),(0.03,0.98,1)):
    e=ArmEnv()
    e._set_dist(d); e._set_angle1(a); e._set_still_count(s)
    ok=0; steps=[]
    for ep in range(30):
        obs,_=e.reset()
        for t in range(e.max_steps+e.max_wait_steps+10):
            act,_=m.predict(obs,deterministic=True)
            obs,r,term,tr,info=e.step(act)
            if e.phase>=3 or term or tr: break
        ok+=int(e.phase>=3); steps.append(t+1)
    print("%.2f/%.2f -> %2d/30, 平均 %.0f 步" % (d,a,ok,np.mean(steps)))

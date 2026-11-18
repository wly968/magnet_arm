import sys, os, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load(r"scene_magnet/saves/phase1.zip", device="cpu")
print("你的 phase1.zip 阈值阶梯（phase1 不终止模式）：")
for d,a,s in ((0.06,0.90,0),(0.05,0.94,0),(0.04,0.95,1),(0.03,0.98,1)):
    e=ArmEnv(terminate_on_phase1=False, terminate_on_phase2=True, max_phase=3)
    e._set_dist(d); e._set_angle1(a); e._set_still_count(s)
    ok=0
    for ep in range(50):
        obs,_=e.reset()
        for t in range(e.max_steps+e.max_wait_steps+10):
            act,_=m.predict(obs,deterministic=True)
            obs,r,term,tr,info=e.step(act)
            if term or tr: break
        ok+=int(e.phase>=3)
    print("  %.2f/%.2f/still%d -> %2d/50" % (d,a,s,ok))

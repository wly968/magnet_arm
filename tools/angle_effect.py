import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
TH1=(0.050,0.940,0)
def run(ang, n=60):
    e=ArmEnv(); ok=0; ys=[]
    for ep in range(n):
        obs,_=e.reset(); y0=float(e.box_pos[1])
        d,a,s=TH1; e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
        last=e.phase
        for t in range(e.max_steps+e.max_wait_steps+10):
            if e.phase!=last:
                last=e.phase
                d,a,s=(0.032,ang,1) if e.phase==2 else TH1
                e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
            act,_=m1.predict(obs,deterministic=True)
            obs,r,term,tr,info=e.step(act)
            if e.phase>=3 or term or tr: break
        ok+=int(e.phase>=3); ys.append((y0, int(e.phase>=3)))
    far=[o for y,o in ys if y>0.70]; near=[o for y,o in ys if y<=0.70]
    return ok, (sum(far), len(far)), (sum(near), len(near))
for ang in (0.980, 0.965, 0.960, 0.945):
    ok, far, near = run(ang)
    print("angle>%.3f : 总 %2d/60 = %3.0f%% | y<=0.70 %d/%d | y>0.70 %d/%d"
          % (ang, ok, 100*ok/60, near[0], near[1], far[0], far[1]))

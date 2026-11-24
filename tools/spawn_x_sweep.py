import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
TH1=(0.050,0.940,0); TH2=(0.032,0.960,1)
e=ArmEnv(); e.spawn_mode="uniform"
def run(x, n=15):
    e.spawn_x_range=(x-1e-4, x+1e-4); ok=0; grabspot=[]
    for ep in range(n):
        obs,_=e.reset()
        d,a,s=TH1; e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
        last=e.phase
        for t in range(e.max_steps+e.max_wait_steps+10):
            if e.phase!=last:
                last=e.phase
                d,a,s=TH2 if e.phase==2 else TH1
                e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
            act,_=m1.predict(obs,deterministic=True)
            obs,r,term,tr,info=e.step(act)
            if e.phase>=3 or term or tr: break
        ok+=int(e.phase>=3)
        if e.phase>=3: grabspot.append(float(e.box_pos[0]))
    return ok, (np.mean(grabspot) if grabspot else float("nan"))
print("按初始 x 强制刷新（带速 0.2 m/s 持续移动），15 局/点")
for x in np.arange(-0.70, -0.29, 0.05):
    ok,g=run(float(x))
    print("  x0=%+.2f : 成功 %2d/15 = %3.0f%% | 实际抓取点 x≈%s" % (x, ok, 100*ok/15, ("%.2f"%g) if g==g else "-"))

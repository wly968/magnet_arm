import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
TH2=(0.032,0.960,1)
def run(p1d, p1a, n=40):
    e=ArmEnv(); e.spawn_mode="uniform"; e.spawn_x_range=(-0.700,-0.420)
    e.phase1_dist, e.phase1_angle = p1d, p1a
    ok=0; p1steps=[]; minang=[]
    for ep in range(n):
        obs,_=e.reset()
        d,a,s=(p1d,p1a,0); e._set_dist(d); e._set_angle1(a); e._set_still_count(s)
        last=e.phase; t1=None; ma=1.0
        for t in range(e.max_steps+e.max_wait_steps+10):
            if e.phase!=last:
                last=e.phase
                if e.phase==2:
                    d,a,s=TH2; e._set_dist(d); e._set_angle1(a); e._set_still_count(s)
                e.phase1_dist, e.phase1_angle = p1d, p1a
            act,_=m1.predict(obs,deterministic=True)
            obs,r,term,tr,info=e.step(act)
            if e.phase==1:
                ee=e.data.site_xpos[e.ee_id]; j3=e.data.xanchor[e.arm_joints[3]]
                f=(ee-j3)/np.linalg.norm(ee-j3); ma=min(ma,float(np.dot(f,[0,0,-1])))
                if t1 is None and e.phase==1: t1=t
            if e.phase==2 and p1steps is not None:
                pass
            if e.phase>=3 or term or tr: break
        ok+=int(e.phase>=3)
        if e.phase>=2: p1steps.append(t+1); minang.append(ma)
    return ok, (np.mean(p1steps) if p1steps else float("nan")), (np.median(minang) if minang else float("nan"))
for p1d,p1a in ((0.050,0.940),(0.045,0.950),(0.042,0.958)):
    ok, steps, ma = run(p1d,p1a)
    print("phase1 %.3f/%.3f -> 成功 %2d/40 | 进入phase2时用步数 %.0f | phase1 内最小 angle %.3f"
          % (p1d,p1a,ok,steps,ma))

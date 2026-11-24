import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
TH1=(0.050,0.940,0); TH2=(0.032,0.980,1); TH35=(0.034,0.980,2); TH6=(0.060,0.940,0)
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
m2=PPO.load("scene_magnet/saves/phase34.zip", device="cpu")
m3=PPO.load("scene_magnet/saves/phase6.zip", device="cpu")
print("dims:", m1.observation_space.shape, m2.observation_space.shape, m3.observation_space.shape)
def pol(ph): return m1 if ph<3 else (m2 if ph<6 else m3)
def adapt(o,m):
    d=m.observation_space.shape[0]; o=np.asarray(o,dtype=np.float32)
    return o[:d] if o.shape[-1]>d else (np.concatenate([o,np.zeros(d-o.shape[-1],dtype=np.float32)]) if o.shape[-1]<d else o)
for max_steps in (200, 400):
    e=ArmEnv(); e.max_steps=max_steps; cnt={}; lens=[]; coll=0
    for ep in range(30):
        obs,_=e.reset(); last=e.phase
        d,a,s=TH1; e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
        for t in range(e.max_steps+e.max_wait_steps+10):
            if e.phase!=last:
                last=e.phase
                d,a,s = TH2 if e.phase==2 else (TH35 if e.phase<6 else TH6)
                e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
            m=pol(e.phase); act,_=m.predict(adapt(obs,m),deterministic=True)
            obs,r,term,tr,info=e.step(act)
            if term or tr: break
        cnt[int(e.phase)]=cnt.get(int(e.phase),0)+1; lens.append(t+1); coll+=int(e.collected)
    print("max_steps=%-4d 分布 %s | 平均原始步数 %.0f | collected %d/30" % (max_steps, dict(sorted(cnt.items())), np.mean(lens), coll))

import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
TH1=(0.050,0.940,0); TH2=(0.032,0.980,1); TH35=(0.034,0.980,2); TH6=(0.060,0.940,0)
m1=PPO.load("scene_magnet/saves/phase2.zip", device="cpu")
m2=PPO.load("scene_magnet/saves/phase34.zip", device="cpu")
m3=PPO.load("scene_magnet/saves/phase6_home.zip", device="cpu")
def pol(ph): return m1 if ph<3 else (m2 if ph<6 else m3)
def adapt(o,m):
    d=m.observation_space.shape[0]; o=np.asarray(o,dtype=np.float32)
    return o[:d] if o.shape[-1]>d else (np.concatenate([o,np.zeros(d-o.shape[-1],dtype=np.float32)]) if o.shape[-1]<d else o)
def chain(n=40, use_all=True):
    e=ArmEnv(); e.terminate_on_grasp=False; e.max_steps=400; cnt={}
    for ep in range(n):
        obs,_=e.reset(); last=e.phase
        d,a,s=TH1; e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
        for t in range(e.max_steps+e.max_wait_steps+10):
            if e.phase!=last:
                last=e.phase
                d,a,s = TH2 if e.phase==2 else (TH35 if e.phase<6 else TH6)
                e._set_dist(d); e._set_angle1(a); e._set_still_count(s); e.phase1_dist,e.phase1_angle=TH1[0],TH1[1]
            m=pol(e.phase) if use_all else m1
            act,_=m.predict(adapt(obs,m),deterministic=True)
            obs,r,term,tr,info=e.step(act)
            if term or tr: break
        cnt[int(e.phase)]=cnt.get(int(e.phase),0)+1
    return cnt
print("全链路 40 局:", dict(sorted(chain(40,True).items())))
print("仅 model1 40 局:", dict(sorted(chain(40,False).items())))

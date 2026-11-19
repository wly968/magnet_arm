import os, sys, time, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
import mujoco
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m1=PPO.load('scene_magnet/saves/phase2.zip', device='cpu')

def handoff(skip_phase0, budget, trials=10):
    ok=0; steps=[]
    e=ArmEnv()
    for _ in range(trials):
        obs,_=e.reset()
        if skip_phase0:
            y=float(e.box_pos[1])
            pos=np.array([e.arrive_x, y, 0.375])
            e.data.mocap_pos[e.mocap_id]=pos
            e.data.mocap_quat[e.mocap_id]=[1,0,0,0]
            e.box_pos=pos.copy(); e.phase=1
            mujoco.mj_forward(e.model,e.data)
            obs=e._get_obs()
        n=0
        for _ in range(budget):
            n+=1
            a,_=m1.predict(obs,deterministic=True)
            obs,r,term,tr,info=e.step(a)
            if e.phase>=3 or term or tr: break
        ok+=int(e.phase>=3); steps.append(n)
    return ok, steps

for name,skip,budget in (("现状：不跳 phase0, 250 步",False,250),
                         ("加预算：不跳 phase0, 500 步",False,500),
                         ("跳 phase0, 80 步",True,80),
                         ("跳 phase0, 150 步",True,150)):
    ok,steps=handoff(skip,budget)
    print("%-26s -> 成功 %2d/10, 中位步数 %.0f, 最大 %d" % (name, ok, np.median(steps), max(steps)))

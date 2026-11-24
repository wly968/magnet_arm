import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load('scene_magnet/ppo_arm_reach.zip', device='cpu')
print("obs", m.observation_space.shape, "| 训练步数", getattr(m,'num_timesteps',None))
print("动作标准差", np.round(np.exp(m.policy.log_std.detach().numpy()),3))

def run(dist,angle,still,rel,episodes=30):
    e=ArmEnv(); e._set_dist(dist); e._set_angle1(angle); e._set_still_count(still); e._set_rel(rel)
    ok=0; mins=[]; npos=[]; n5=[]
    for ep in range(episodes):
        obs,_=e.reset(); md=9; pos=0; five=0
        for t in range(e.max_steps+e.max_wait_steps+10):
            a,_=m.predict(obs,deterministic=True)
            obs,r,term,tr,info=e.step(a)
            if e.phase==2:
                ee=e.data.site_xpos[e.ee_id]; tgt=e.box_pos+e.grasp_offset
                d=float(np.linalg.norm(ee-tgt)); md=min(md,d)
                j3=e.data.xanchor[e.arm_joints[3]]; f=(ee-j3)/np.linalg.norm(ee-j3)
                a1=float(np.dot(f,[0,0,-1])); vr=float(np.linalg.norm(e.ee_val-e.box_vel))
                if d<dist and a1>angle and vr<rel: pos+=1; five+=5
            if e.phase>=3 or term or tr: break
        ok+=int(e.phase>=3); mins.append(md); npos.append(pos); n5.append(five)
    return ok, np.median(mins), np.median(npos)

for (d,a,s,rv) in ((0.03,0.99,2,0.05),(0.03,0.98,0,0.05),(0.05,0.94,0,0.10)):
    ok,md,np_=run(d,a,s,rv)
    print("dist %.2f/angle %.2f/still %d/rel %.2f -> 成功 %2d/30, 最小dist 中位 %.3f, 在位帧中位 %.0f" % (d,a,s,rv,ok,md,np_))

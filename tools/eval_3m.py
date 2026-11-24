import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m=PPO.load('scene_magnet/saves/phase2.zip', device='cpu')
print("obs", m.observation_space.shape, "| 训练步数", getattr(m,'num_timesteps',None))
print("动作标准差", np.round(np.exp(m.policy.log_std.detach().numpy()),3))

def ev(d,a,st,rv,n=30):
    e=ArmEnv(); e._set_dist(d); e._set_angle1(a); e._set_still_count(st); e._set_rel(rv)
    rel=[]; relpos=[]; ok=0; pos=0; tot=0; runmax=[]
    for ep in range(n):
        obs,_=e.reset(); run=0; best=0
        for t in range(e.max_steps+e.max_wait_steps+10):
            act,_=m.predict(obs,deterministic=True)
            obs,r,term,tr,info=e.step(act)
            if e.phase==2:
                ee=e.data.site_xpos[e.ee_id]; tgt=e.box_pos+e.grasp_offset
                dd=float(np.linalg.norm(ee-tgt))
                j3=e.data.xanchor[e.arm_joints[3]]; f=(ee-j3)/np.linalg.norm(ee-j3)
                aa=float(np.dot(f,[0,0,-1])); vv=float(np.linalg.norm(e.ee_val-e.box_vel))
                band=(ee[2]<tgt[2]+0.03) and (ee[2]>e.box_pos[2]+0.03)
                rel.append(vv); tot+=1
                if dd<d and aa>a and band: pos+=1; relpos.append(vv)
                good = dd<d and aa>a and band and vv<rv
                run = run+1 if good else 0; best=max(best,run)
            if e.phase>=3 or term or tr: break
        ok+=int(e.phase>=3); runmax.append(best)
    rel=np.array(rel); relpos=np.array(relpos) if relpos else np.array([np.nan])
    print("  阈 %.3f/%.2f/still%d/rel%.2f -> 成功 %2d/%d | 位置到位 %4.1f%% | 到位时 rel 中位 %.3f | 总体 rel 中位 %.3f | 最长连续达标 中位 %d" %
          (d,a,st,rv,ok,n,100*pos/max(1,tot),np.nanmedian(relpos),np.median(rel),int(np.median(runmax))))

print("当前课程档位（3M 步时应处于第 3 档）:")
ev(0.042,0.958,1,0.10)
print("其他档位对照:")
ev(0.050,0.940,0,0.20)
ev(0.030,0.990,2,0.05)

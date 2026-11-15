import os, sys, time, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"
torch.set_num_threads(1)
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from envs.arm_env import ArmEnv, CurriculumCallback

env=DummyVecEnv([lambda: Monitor(ArmEnv()) for _ in range(24)])
m=PPO.load(os.path.join(SCENE,"saves","phase1.zip"), env=env, device="cpu", learning_rate=1e-4)
m.ent_coef = 0.02                      # 提高探索
with torch.no_grad():
    m.policy.log_std.fill_(-0.7)       # std≈0.5，重新打开探索
print("初始 std", np.round(np.exp(m.policy.log_std.detach().numpy()),2))
CurriculumCallback.SCHEDULE=[(0,0.050,0.940,0),(700_000,0.042,0.950,1),(1_400_000,0.035,0.960,1)]
t0=time.time(); m.learn(total_timesteps=3_000_000, callback=CurriculumCallback()); print("3M 训练 %.0fs"%(time.time()-t0))
m.save(os.path.join(SCENE,"saves","phase12_explore.zip"))
print("结束 std", np.round(np.exp(m.policy.log_std.detach().numpy()),2))
for d_,a_,s_ in ((0.035,0.96,1),(0.040,0.96,1),(0.035,0.97,1)):
    e=ArmEnv(); e.max_steps=250
    e._set_dist(d_); e._set_angle1(a_); e._set_still_count(s_)
    ok=0; steps=[]
    for ep in range(150):
        o,_=e.reset()
        for t in range(e.max_steps):
            act,_=m.predict(o,deterministic=True)
            o,r,term,tr,info=e.step(act)
            if term or tr: break
        ok+=int(e.phase>=3); steps.append(t+1)
    print("  %.3f/%.2f/%d -> %3d/150 = %5.1f%%, 平均 %.1f 步"%(d_,a_,s_,ok,100*ok/150,np.mean(steps)))

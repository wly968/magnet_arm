import os, sys, time, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from envs.arm_env import ArmEnv
from envs.phase34 import Phase34Env

m1=PPO.load('scene_magnet/saves/phase2.zip', device='cpu')

# 1) 裸环境 fps
env=DummyVecEnv([lambda: Monitor(ArmEnv()) for _ in range(24)])
env.reset(); t0=time.time()
for _ in range(2000): env.step(np.zeros((24,4)))
raw=2000/(time.time()-t0)

# 2) 交接环境：统计每次 reset 的步数
class CountEnv(Phase34Env):
    def __init__(self,*a,**k):
        super().__init__(*a,**k); self.handoff_steps=0; self.tries=0; self.fallback=False
    def _run_handoff(self):
        self.handoff_steps=0; self.tries=0; self.fallback=False
        for k in range(self.attempts):
            self.tries+=1
            obs,info=self.env.reset()
            for _ in range(self.max_handoff_steps):
                self.handoff_steps+=1
                a,_=self.model1.predict(obs,deterministic=True)
                obs,r,term,trunc,info=self.env.step(a)
                if self.env.unwrapped.phase>=3 or term or trunc: break
            if self.env.unwrapped.phase>=3: return obs,{}
        self.fallback=True
        self.env.unwrapped.phase=3
        return self.env.unwrapped._get_obs(),{}

envs=[CountEnv(ArmEnv(), m1) for _ in range(24)]
venv=DummyVecEnv([lambda e=e: e for e in envs])
t0=time.time(); venv.reset(); t_r=time.time()-t0
stats=[(e.handoff_steps,e.tries,e.fallback) for e in envs]
hs=np.array([s[0] for s in stats]); tr=np.array([s[1] for s in stats]); fb=np.array([s[2] for s in stats])
print("裸环境 fps: %.0f" % raw)
print("交接 reset: 24 个环境, 用时 %.2fs, 每 env 交接步数 中位 %.0f 最大 %d, 尝试次数 中位 %.0f, 兜底比例 %.0f%%"
      % (t_r, np.median(hs), hs.max(), np.median(tr), 100*fb.mean()))
print("平均每个 reset 的交接步数: %.0f" % hs.mean())

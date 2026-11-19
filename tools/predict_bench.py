import os, sys, time, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
m1=PPO.load('scene_magnet/saves/phase2.zip', device='cpu')
e=ArmEnv(); obs,_=e.reset()
obs=np.asarray(obs, dtype=np.float32)
t0=time.perf_counter()
for _ in range(1000): m1.predict(obs, deterministic=True)
print("m1.predict 单帧: %.0f us" % ((time.perf_counter()-t0)/1000*1e6))
t0=time.perf_counter()
for _ in range(1000): e.step(np.zeros(4))
print("env.step       : %.0f us" % ((time.perf_counter()-t0)/1000*1e6))
print("=> 交接每步成本约 %.0f us，750 步约 %.0f ms" % (
    ((time.perf_counter()-t0)/1000*1e6)*0, 0))

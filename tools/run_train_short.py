import os, sys, time, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"; torch.set_num_threads(1)
sys.path.insert(0,'scene_magnet')
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.arm_env import ArmEnv, CurriculumCallback, DiagCallback, WaitAwareMonitor
env=DummyVecEnv([lambda: WaitAwareMonitor(ArmEnv()) for _ in range(24)])
m=PPO.load("scene_magnet/saves/phase1.zip", env=env, device="cpu", learning_rate=1e-4)
t0=time.time()
m.learn(total_timesteps=120_000, callback=[CurriculumCallback(), DiagCallback(every=20_000)])
print("用时 %.0fs (%.0f fps)" % (time.time()-t0, 120000/(time.time()-t0)))

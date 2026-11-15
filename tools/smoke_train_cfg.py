import os, sys
os.environ["OMP_NUM_THREADS"]="1"
import torch; torch.set_num_threads(1)
sys.path.insert(0, r"scene_magnet")
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from envs.arm_env import ArmEnv
import time
vec = DummyVecEnv([lambda: Monitor(ArmEnv()) for _ in range(4)])
m = PPO("MlpPolicy", vec, n_steps=512, batch_size=1024, n_epochs=5, learning_rate=3e-4,
        policy_kwargs=dict(net_arch=[128,128]), target_kl=0.02, device="cpu", verbose=0)
t0=time.time(); m.learn(total_timesteps=8192); dt=time.time()-t0
print("训练配置冒烟测试通过: %.0f fps (4 envs)" % (8192/dt))

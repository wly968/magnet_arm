import os, sys, time, numpy as np, torch
os.environ["OMP_NUM_THREADS"]="1"
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from envs.arm_env import ArmEnv

def run(n_envs, threads, n_epochs, batch, net, steps=20000):
    torch.set_num_threads(threads)
    env=DummyVecEnv([lambda: Monitor(ArmEnv()) for _ in range(n_envs)])
    t0=time.perf_counter()
    m=PPO("MlpPolicy", env, n_steps=512, batch_size=batch, n_epochs=n_epochs,
          learning_rate=3e-4, ent_coef=0.005, device="cpu",
          policy_kwargs=dict(net_arch=net), seed=0, verbose=0, target_kl=0.02)
    m.learn(total_timesteps=steps)
    dt=time.perf_counter()-t0
    return steps/dt

print("n_envs threads epochs batch net      fps")
for cfg in [
    (8, 20, 10, 512, [256,256]),
    (8, 1,  10, 512, [256,256]),
    (16,1,  10, 512, [256,256]),
    (24,1,  10, 512, [256,256]),
    (16,1,   5, 1024,[256,256]),
    (16,1,   5, 1024,[128,128]),
]:
    try:
        fps=run(*cfg)
        print("%2d  %2d   %2d  %4d %-9s %.0f" % (*cfg[:4], str(cfg[4]), fps))
    except Exception as e:
        print(cfg, "FAILED", e)

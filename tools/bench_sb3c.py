import os, sys, time, torch
os.environ["OMP_NUM_THREADS"]="1"
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from envs.arm_env import ArmEnv

def run(n_envs, n_steps, n_epochs, batch, net, steps=24000):
    torch.set_num_threads(1)
    env=DummyVecEnv([lambda: Monitor(ArmEnv()) for _ in range(n_envs)])
    t0=time.perf_counter()
    m=PPO("MlpPolicy", env, n_steps=n_steps, batch_size=batch, n_epochs=n_epochs,
          learning_rate=3e-4, ent_coef=0.005, device="cpu",
          policy_kwargs=dict(net_arch=net), seed=0, verbose=0, target_kl=0.02)
    m.learn(total_timesteps=steps)
    return steps/(time.perf_counter()-t0)

print("n_envs n_steps epochs batch net        fps")
for cfg in [
    (16, 512, 3, 1024, [64,64]),
    (16, 512, 5, 2048, [64,64]),
    (24, 512, 5, 1024, [64,64]),
    (32, 512, 5, 1024, [64,64]),
    (16, 512, 5, 1024, [32,32]),
    (12, 512, 5, 1024, [64,64]),
]:
    try:
        print("%2d %5d %2d %5d %-11s %.0f" % (*cfg[:4], str(cfg[4]), run(*cfg)))
    except Exception as e:
        print(cfg, "FAILED", type(e).__name__)

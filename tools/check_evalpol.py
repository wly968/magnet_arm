import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from envs.arm_env import ArmEnv

m=PPO.load(os.path.join(SCENE,"saves","arm4dof_10m.zip"), device="cpu")
ev = DummyVecEnv([lambda: Monitor(ArmEnv())])
ev.env_method("_set_dist", 0.03); ev.env_method("_set_angle1", 0.96); ev.env_method("_set_still_count", 2)
mean_r, std_r = evaluate_policy(m, ev, n_eval_episodes=20, deterministic=True, return_episode_rewards=False)
print("evaluate_policy: %.2f +/- %.2f" % (mean_r, std_r))
ph = ev.envs[0].unwrapped.phase
print("最后一个 episode 结束 phase:", ph)

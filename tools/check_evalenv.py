import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from envs.arm_env import ArmEnv

m=PPO.load(os.path.join(SCENE,"saves","arm4dof_10m.zip"), device="cpu")

ev = DummyVecEnv([lambda: Monitor(ArmEnv())])
ev.env_method("_set_dist", 0.03); ev.env_method("_set_angle1", 0.96); ev.env_method("_set_still_count", 2)
print("阈值:", ev.get_attr("dist_threshold"), ev.get_attr("angle1_threshold"), ev.get_attr("still_count_threshold"))
print("step 返回格式检查:")
print(" obs shape", ev.reset()[0].shape, "obs_space", ev.observation_space.shape)

# 用 vec env 跑 20 局
ok=0
for ep in range(20):
    o = ev.reset()
    for t in range(250):
        a,_ = m.predict(o, deterministic=True)
        o,r,done,info = ev.step(a)
        if done[0]: break
    ok += int(ev.envs[0].unwrapped.phase>=3) if False else 0
print("done")

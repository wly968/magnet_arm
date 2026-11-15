import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv()
print("obs_space", env.observation_space.shape, "act_space", env.action_space.shape)
model=PPO.load(os.path.join(SCENE,"saves","best_model.zip"), device="cpu")
print("model act_space", model.action_space.shape)
for ep in range(5):
    o,_=env.reset(seed=ep); weld_steps=0; minz=9; ok=False
    for t in range(env.max_steps):
        a,_=model.predict(o,deterministic=True)
        o,r,term,tr,info=env.step(a)
        w = int(env.data.eq_active[env.weld_id]) if env.weld_id is not None else 0
        weld_steps += w
        minz=min(minz, float(env.data.site_xpos[env.ee_id][2]))
        ok = ok or bool(info.get("success"))
        if term or tr: break
    print("ep%d success=%s steps=%d weld_steps=%d ee_min_z=%.3f final_box_z=%.3f" % (ep, ok, t+1, weld_steps, minz, env.box_pos[2]))

import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
p=os.path.join(SCENE,"ppo_arm_reach.zip")
m=PPO.load(p, device="cpu")
print("model obs", m.observation_space.shape, "act", m.action_space.shape)

for name,(d_,a_,s_) in {"500k档":(0.06,0.90,0),"1M档":(0.025,0.98,2)}.items():
    env=ArmEnv(); env.max_steps=250
    env._set_dist(d_); env._set_angle1(a_); env._set_still_count(s_)
    ok=0; steps=[]; hit=0; best=9
    for ep in range(200):
        o,_=env.reset(); had=False
        for t in range(env.max_steps):
            a,_=m.predict(o,deterministic=True)
            o,r,term,tr,info=env.step(a)
            if env.phase==2:
                ee=env.data.site_xpos[env.ee_id]; tgt=env.box_pos+env.grasp_offset
                best=min(best,float(np.linalg.norm(ee-tgt)))
            if env.cube_hit: had=True
            if term or tr: break
        ok+=int(env.phase>=3); steps.append(t+1); hit+=int(had)
    print("%s -> phase2完成 %d/200, 平均步数 %.1f, 有碰撞局 %d, best dist %.4f"
          % (name, ok, np.mean(steps), hit, best))

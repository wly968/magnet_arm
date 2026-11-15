import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv()
path=os.path.join(SCENE,"saves","phase2.zip")
model=PPO.load(path, env=env)
print("loaded:", path)
print("obs space", model.observation_space.shape, "action space", model.action_space.shape)
print("net", model.policy.mlp_extractor.policy_net, model.policy.mlp_extractor.value_net)
print("ent_coef", model.ent_coef, "lr", model.learning_rate)
N=60; ph1=0; ph2=0; ph3=0; ph4=0; q2=[]; a2=[]; d2=[]; s2=[]; ends={}
for ep in range(N):
    obs,_=env.reset(); term=False; maxph=1; qs=[]; a2s=[]; ds=[]; ms=0
    for t in range(env.max_steps):
        a,_=model.predict(obs,deterministic=True)
        obs,r,term,trunc,info=env.step(a)
        maxph=max(maxph,env.phase); ms=max(ms,env.still_count)
        if env.phase==2:
            d=env.data; qs.append(float(d.qpos[env.finger_qpos_adr]))
            rot=d.geom_xmat[env.box_geom_id].reshape(3,3); px=d.geom_xmat[env.palm_id].reshape(3,3)
            a2s.append(abs(float(np.dot(px[:2,1]/np.linalg.norm(px[:2,1]), rot[:2,0]/np.linalg.norm(rot[:2,0])))))
            ee=d.site_xpos[env.ee_id]; ds.append(float(np.linalg.norm(ee-(env.box_pos+np.array([0,0,0.15])))))
        if term or trunc: break
    ends[maxph]=ends.get(maxph,0)+1
    ph1+=int(maxph>=2); ph2+=int(maxph>=3); ph3+=int(maxph>=4)
    if qs: q2.append(np.mean(qs)); a2.append(np.mean(a2s)); d2.append(np.mean(ds)); s2.append(ms)
print(f"phase2 entry {ph1}/{N}; phase2 success {ph2}/{N}; phase3 {ph3}; phase4 {ph4}")
print("end phases", ends)
print("phase2 q", round(float(np.mean(q2)),4), "angle2", round(float(np.mean(a2)),3),
      "dist", round(float(np.mean(d2)),3), "still", round(float(np.mean(s2)),1))

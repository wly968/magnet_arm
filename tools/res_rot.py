import os, sys, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
env=ArmEnv(); m=PPO.load(os.path.join(SCENE,"saves","phase2.zip"), env=env)
def run(angle_deg):
    N=30; succ=0; steps=[]; ees=[]; qs=[]; ph={}
    for ep in range(N):
        obs,_=env.reset()
        a=env.box_qpos_adr; ang=np.deg2rad(angle_deg)
        env.data.qpos[a:a+7]=[0,0.6,0.375,np.cos(ang/2),0,0,np.sin(ang/2)]
        env.data.qvel[env.box_qvel_adr:env.box_qvel_adr+6]=0
        env.box_pos=np.array([0,0.6,0.375]); mujoco.mj_forward(env.model,env.data); obs=env._get_obs()
        term=False
        for t in range(env.max_steps):
            act,_=m.predict(obs,deterministic=True)
            obs,r,term,trunc,info=env.step(act)
            if env.phase==3:
                ees.append(float(env.data.site_xpos[env.ee_id][2])); qs.append(float(env.data.qpos[env.finger_qpos_adr]))
            if term or trunc: break
        if term and env.phase==4: succ+=1
        ph[env.phase]=ph.get(env.phase,0)+1; steps.append(t+1)
    ez = f"{np.mean(ees):.3f}" if ees else "-"
    qz = f"{np.mean(qs):.4f}" if qs else "-"
    print(f"rot={angle_deg:3d} success {succ}/{N} steps {np.mean(steps):5.0f} ee_z {ez} q {qz} end {ph}")
for ang in (0, 15, 30, 45, 60, 90):
    run(ang)

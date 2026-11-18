import os, sys, numpy as np, mujoco
os.environ["OMP_NUM_THREADS"]="1"
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from envs.arm_env import ArmEnv
from envs.phase34 import Phase34Env
m1=PPO.load(os.path.join(SCENE,"ppo_arm_reach.zip"), device="cpu")
e=ArmEnv(); e.max_steps=250
e._set_dist(0.05); e._set_angle1(0.945); e._set_still_count(0)
env=Phase34Env(e, m1); env.reset()
q0=e.data.qpos[e.arm_qpos_adr].copy()
base=(q0-e.home_ctrl)/e.ctrl_scale
print("抓取姿态 q", np.round(q0,2), "ee_z %.3f"%e.data.site_xpos[e.ee_id][2])
for dsh in (-0.2, 0.2):
    e2=ArmEnv(); e2.reset(); e2.data.qpos[e2.arm_qpos_adr]=q0; e2.data.qvel[:]=0; mujoco.mj_forward(e2.model,e2.data)
    a=np.zeros(4); a[1]=dsh
    for _ in range(20): e2.step(a)
    print("shoulder %+.2f -> ee_z %.3f" % (dsh, e2.data.site_xpos[e2.ee_id][2]))

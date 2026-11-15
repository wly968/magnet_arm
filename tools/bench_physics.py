import os, sys, time, numpy as np, mujoco
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
m=mujoco.MjModel.from_xml_path(os.path.join(SCENE,"assets","models","train_scene.xml"))
d=mujoco.MjData(m)

def bench(fn, n=3000):
    fn(10)  # warmup
    t=time.perf_counter()
    fn(n)
    return n/ (time.perf_counter()-t)

print("physics loop 5  : %.0f mj_step/s" % bench(lambda n: [mujoco.mj_step(m,d) for _ in range(n)]))
print("physics nstep=5 : %.0f mj_step/s" % bench(lambda n: [mujoco.mj_step(m,d,5) for _ in range(n//5)]))

from envs.arm_env import ArmEnv
env=ArmEnv(); env.reset()
a=np.zeros(env.action_space.shape)
def env_steps(n):
    for _ in range(n): env.step(a)
print("env.step (5 mj_step, python loop): %.0f env-steps/s" % bench(env_steps, 1000))

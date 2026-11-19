import os, sys, time, numpy as np, mujoco
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
from envs.arm_env import ArmEnv

m=mujoco.MjModel.from_xml_path('scene_magnet/assets/models/train_scene.xml')
d=mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m,d,0); mujoco.mj_forward(m,d)

def bench(fn, n=2000):
    fn(); t0=time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter()-t0)/n*1e6      # 微秒

print("mj_step(nstep=5)        %8.1f us" % bench(lambda: mujoco.mj_step(m,d,5)))
res=np.zeros(6); sid=m.site('ee_site').id
print("mj_objectVelocity       %8.1f us" % bench(lambda: mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_SITE,sid,res,0)))
print("mj_forward              %8.1f us" % bench(lambda: mujoco.mj_forward(m,d)))

e=ArmEnv(); e.reset(); a=np.zeros(4)
print("env.step (含上面全部)    %8.1f us" % bench(lambda: e.step(a), 1000))
print("env._get_obs            %8.1f us" % bench(lambda: e._get_obs(), 2000))
print("-> 单 env 理论 fps: %.0f" % (1e6/bench(lambda: e.step(a), 500)))

import os, sys, inspect
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from envs.arm_env import ArmEnv
print("module file:", inspect.getfile(ArmEnv))
src = inspect.getsource(ArmEnv.step)
print(src[:2600])

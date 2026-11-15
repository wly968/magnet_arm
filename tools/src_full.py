import os, sys, inspect
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene"); sys.path.insert(0,SCENE)
from envs.arm_env import ArmEnv
print(inspect.getsource(ArmEnv.step))

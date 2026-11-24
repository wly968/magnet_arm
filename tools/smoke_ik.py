import sys, time, numpy as np
sys.path.insert(0,'scene_magnet')
from envs.arm_env import ArmEnv
e=ArmEnv(); o,_=e.reset()
print("obs", o.shape, "| 末尾4维(不确定性)", o[-4:])
# 用零动作走完 phase0 -> 看 phase1/2 时 IK 是否把末端带到目标
t0=time.perf_counter(); n=0
while e.phase<3 and n<600:
    o,r,term,tr,info=e.step(np.zeros(4)); n+=1
    if e.phase==2 and n%10==0:
        ee=e.data.site_xpos[e.ee_id]; tgt=e._target()
        print("  step%3d phase=%d dist=%.4f" % (n, e.phase, float(np.linalg.norm(ee-tgt))))
    if term or tr: break
print("结束 phase", e.phase, "| 用时 %.0f us/步" % ((time.perf_counter()-t0)/max(1,n)*1e6))

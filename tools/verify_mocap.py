import sys, os, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
import test as t
env=t.ArmEnv(terminate_on_phase2=False)
models=t.load_models(env)
m1,m2,m3=models
obs,_=env.reset(); t.apply_thresholds(env, env.phase); last=env.phase
carry=[]; rel=None
for i in range(400):
    if env.phase!=last: t.apply_thresholds(env, env.phase)
    last=env.phase
    pol=t.pick_policy(models, env.phase)
    a,_=pol.predict(obs,deterministic=True)
    obs,r,term,tr,info=env.step(a)
    if env.phase in (3,4,5):
        ee=env.data.site_xpos[env.ee_id]
        carry.append((float(ee[2])-float(env.box_pos[2]),
                      tuple(np.round(env.data.mocap_quat[env.mocap_id],3))))
    if term or tr: break
print("carry 样本 %d, 末端-方块中心 dz 均值 %.3f (应 0.060)" % (len(carry), np.mean([c[0] for c in carry])))
print("carry 期间 quat 唯一值:", set(c[1] for c in carry))
print("结束后 phase=%d, 方块最终位置 %s (期望 0.65,-0.45,0.375)" % (env.phase, np.round(env.box_pos,3)))

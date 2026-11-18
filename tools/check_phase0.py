import sys, os, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
import test as t
env=t.ArmEnv(); env.smooth_alpha=0.5
models=t.load_models(env)
ph={}; steps=[]; attach=[]
for ep in range(50):
    obs,_=env.reset(); t.apply_thresholds(env, env.phase); last=env.phase
    for i in range(env.max_steps+env.max_wait_steps+10):
        if env.phase!=last:
            t.apply_thresholds(env, env.phase, entry=True); last=env.phase
        pol=t.pick_policy(models, env.phase)
        a,_=pol.predict(obs,deterministic=True)
        prev=env.phase
        obs,r,term,tr,info=env.step(a)
        if prev==2 and env.phase==3:
            ee=env.data.site_xpos[env.ee_id]; attach.append(float(np.linalg.norm(ee-(env.box_pos+env.grasp_offset))))
        if term or tr: break
    ph[env.phase]=ph.get(env.phase,0)+1; steps.append(i+1)
print("50 局:", ph, "平均步数 %.0f" % np.mean(steps))
print("抓取瞬间末端-目标 距离: 均值 %.3f m, 最大 %.3f m (阈值 %.3f)" % (np.mean(attach), np.max(attach), 0.03))

import sys, os, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
import test as t
env=t.ArmEnv(); env.smooth_alpha=0.5
models=t.load_models(env)
for ep in range(5):
    obs,_=env.reset(); t.apply_thresholds(env,env.phase); last=env.phase
    for i in range(1200):
        if env.phase!=last:
            t.apply_thresholds(env,env.phase,entry=True); last=env.phase
        pol=t.pick_policy(models, env.phase); a,_=pol.predict(obs,deterministic=True)
        obs,r,term,tr,info=env.step(a)
        if env.phase>=3 or term or tr: break
    print("ep%d phase=%d steps=%d term=%s trunc=%s nan=%s" % (ep, env.phase, i+1, term, tr, info.get("nan")))

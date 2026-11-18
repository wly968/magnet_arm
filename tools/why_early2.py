import sys, os, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scene_magnet')
import test as t
env=t.ArmEnv(); env.smooth_alpha=0.5
m1=t.load_model(t.find_model("model1"), 19, "model1")
m2=t.load_model(t.find_model("model2"), 19, "model2")
m3=t.load_model(t.find_model("model3"), 19, "model3")
models=[m1,m2,m3]
for ep in range(5):
    obs,_=env.reset(); t.apply_thresholds(env,env.phase); last=env.phase
    for i in range(1200):
        if env.phase!=last:
            t.apply_thresholds(env,env.phase,entry=True); last=env.phase
        pol=t.pick_policy(models, env.phase); a,_=pol.predict(obs,deterministic=True)
        obs,r,term,tr,info=env.step(a)
        if env.phase>=3 or term or tr: break
    print("ep%d phase=%d steps=%d wait=%d term=%s trunc=%s nan=%s"
          % (ep, env.phase, i+1, env.wait_steps, term, tr, info.get("nan")))

import os, sys, numpy as np
ROOT=r"D:\programming\python\mujoco_xml"; SCENE=os.path.join(ROOT,"scene_magnet"); sys.path.insert(0,SCENE)
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.arm_env import ArmEnv
N=8; STEPS=1_000_000
vec = make_vec_env(ArmEnv, n_envs=N, seed=0, vec_env_cls=DummyVecEnv)
model = PPO("MlpPolicy", vec, verbose=0, learning_rate=3e-4, n_steps=512, batch_size=512,
            n_epochs=10, gamma=0.99, gae_lambda=0.95, ent_coef=0.005, device="cpu",
            policy_kwargs=dict(net_arch=[256,256]), seed=0)
model.learn(total_timesteps=STEPS)
model.save(os.path.join(SCENE,"saves","magnet_5dim.zip"))
env=ArmEnv(); succ=0; rets=[]; lens=[]; mag=[]
for ep in range(20):
    o,_=env.reset(); tot=0.0; ok=False; m=0
    for t in range(env.max_steps):
        a,_=model.predict(o,deterministic=True)
        o,r,term,tr,info=env.step(a); tot+=r
        ok = ok or bool(info.get("success")); m += int(env.magnet_on)
        if term or tr: break
    succ+=int(ok); rets.append(tot); lens.append(t+1); mag.append(m)
print("RESULT 5dim success %d/20 mean-return %.2f mean-len %.1f magnet-on %.1f"
      % (succ, np.mean(rets), np.mean(lens), np.mean(mag)))

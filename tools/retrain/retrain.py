import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCENE = os.path.join(ROOT, "scene")
sys.path.insert(0, SCENE)
os.chdir(SCENE)  # arm_env 用相对路径加载 train_scene.xml

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from arm_env import ArmEnv

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppo_current")

env = make_vec_env(ArmEnv, n_envs=8, seed=0)
model = PPO(
    "MlpPolicy", env, verbose=0,
    learning_rate=3e-4, n_steps=512, batch_size=256, n_epochs=10,
    gamma=0.99, gae_lambda=0.95, ent_coef=0.005, device="cpu",
    policy_kwargs=dict(net_arch=[256, 256]), seed=0,
)
model.learn(total_timesteps=2_000_000, progress_bar=False)
model.save(OUT)
print("SAVED", OUT)

import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCENE = os.path.join(ROOT, "scene")
sys.path.insert(0, SCENE)
os.chdir(SCENE)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from envs.arm_env import ArmEnv, CurriculumCallback

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase3_ft")

if __name__ == "__main__":
    env = make_vec_env(ArmEnv, n_envs=16, seed=0)
    model = PPO.load(os.path.join(SCENE, "saves", "phase2.zip"), env=env)
    model.learn(total_timesteps=500_000, callback=[CurriculumCallback()], progress_bar=False)
    model.save(OUT)
    print("SAVED", OUT)

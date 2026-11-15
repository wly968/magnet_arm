import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCENE = os.path.join(ROOT, "scene")
sys.path.insert(0, SCENE); os.chdir(SCENE)
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from envs.arm_env import ArmEnv, CurriculumCallback, PhaseTrackerCallback

if __name__ == "__main__":
    env = make_vec_env(ArmEnv, n_envs=8, seed=0)
    model = PPO.load(os.path.join(SCENE, "saves", "phase2.zip"), env=env)
    model.verbose = 1
    model.learn(total_timesteps=400_000, callback=[CurriculumCallback(), PhaseTrackerCallback()])
    model.save(os.path.join(ROOT, "tools", "retrain", "explosion_dummy"))
    print("SAVED")

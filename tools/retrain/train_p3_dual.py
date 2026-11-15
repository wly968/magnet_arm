import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCENE = os.path.join(ROOT, "scene")
sys.path.insert(0, SCENE); os.chdir(SCENE)
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from envs.arm_env import ArmEnv, PhaseTrackerCallback

class Phase3Env(gym.Wrapper):
    """只把 reset 改成 phase 3 起始，用于训练第二个（phase3）模型"""
    def __init__(self):
        super().__init__(ArmEnv())
    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, option={"phase": 3})

if __name__ == "__main__":
    env = make_vec_env(Phase3Env, n_envs=8, seed=0)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=256,
                batch_size=256, n_epochs=5, gamma=0.99, gae_lambda=0.95,
                ent_coef=0.001, device="cpu", policy_kwargs=dict(net_arch=[128,128]), seed=0)
    model.learn(total_timesteps=300_000)
    model.save(os.path.join(ROOT, "tools", "retrain", "model2_p3"))
    print("DONE")

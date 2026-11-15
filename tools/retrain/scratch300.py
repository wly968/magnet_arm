import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCENE = os.path.join(ROOT, "scene")
sys.path.insert(0, SCENE); os.chdir(SCENE)
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from envs.arm_env import ArmEnv, CurriculumCallback, PhaseTrackerCallback

if __name__ == "__main__":
    env = make_vec_env(ArmEnv, n_envs=8, seed=0)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=512,
                batch_size=512, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                ent_coef=0.005, device="cpu", policy_kwargs=dict(net_arch=[256,256]), seed=0)
    model.learn(total_timesteps=300_000, callback=[CurriculumCallback(), PhaseTrackerCallback()])
    print("DONE")

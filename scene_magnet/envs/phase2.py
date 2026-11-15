"""phase 2 专用训练环境：每次 reset 先跑 phase1.zip 走到 phase 2，再交棒。

这样所有样本都落在 phase 2 的状态分布里，不会被 phase 1 的任务分散。
"""
import os
import sys

import gymnasium as gym

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Phase2Env(gym.Wrapper):
    def __init__(self, env, model1, max_handoff_steps=250, attempts=3):
        super().__init__(env)
        self.model1 = model1
        self.max_handoff_steps = max_handoff_steps
        self.attempts = attempts

    def _handoff(self):
        for _ in range(self.attempts):
            obs, info = self.env.reset()
            for _ in range(self.max_handoff_steps):
                a, _ = self.model1.predict(obs, deterministic=True)
                obs, r, term, trunc, info = self.env.step(a)
                if self.env.unwrapped.phase >= 2 or term or trunc:
                    break
            if self.env.unwrapped.phase >= 2:
                return obs, {}
        self.env.unwrapped.phase = 2
        return self.env.unwrapped._get_obs(), {}

    def reset(self, **kwargs):
        return self._handoff()

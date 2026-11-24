import os
import sys

import gymnasium as gym
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Phase34Env(gym.Wrapper):
    """phase 3/4 训练环境：每次 reset 先跑 model1(phase1/2)，拿到真实交接姿态。

    - 正常情况：model1 到达 phase 3 后交棒，本条 episode 从该姿态继续；
    - 兜底：model1 连续 3 次都没到 phase 3，直接强制 phase=3。
    """

    def __init__(self, env, model1, max_handoff_steps=250, attempts=3):
        super().__init__(env)
        self.model1 = model1
        self.max_handoff_steps = max_handoff_steps
        self.attempts = attempts

    def _reset_counters(self):
        # 交接用的步数和等待步数都不计入本回合
        u = self.env.unwrapped
        u.steps = 0
        u.wait_steps = 0

    def _run_handoff(self):
        for _ in range(self.attempts):
            obs, info = self.env.reset()
            for _ in range(self.max_handoff_steps):
                a, _ = self.model1.predict(obs, deterministic=True)
                obs, r, term, trunc, info = self.env.step(a)
                if self.env.unwrapped.phase >= 3 or term or trunc:
                    break
            if self.env.unwrapped.phase >= 3:
                self._reset_counters()
                return obs, {}
        # 兜底：三次都没送到 phase 3，直接强设
        self.env.unwrapped.phase = 3
        self._reset_counters()
        return self.env.unwrapped._get_obs(), {}

    def reset(self, **kwargs):
        return self._run_handoff()

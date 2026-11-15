"""phase 6（回初始位置）训练环境：reset 时先跑 model1、model2 走到 phase 6。"""
import os
import sys

import gymnasium as gym

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Phase6Env(gym.Wrapper):
    def __init__(self, env, model1, model2, max_handoff_steps=400, attempts=3):
        super().__init__(env)
        self.model1 = model1
        self.model2 = model2
        self.max_handoff_steps = max_handoff_steps
        self.attempts = attempts

    def _handoff(self):
        for _ in range(self.attempts):
            obs, info = self.env.reset()
            ok = True
            # model1 跑到 phase 3
            for _ in range(self.max_handoff_steps):
                if self.env.unwrapped.phase >= 3:
                    break
                a, _ = self.model1.predict(obs, deterministic=True)
                obs, r, term, trunc, info = self.env.step(a)
                if term or trunc:
                    ok = False
                    break
            if not ok:
                continue
            # model2 跑到 phase 6
            for _ in range(self.max_handoff_steps):
                if self.env.unwrapped.phase >= 6:
                    break
                a, _ = self.model2.predict(obs, deterministic=True)
                obs, r, term, trunc, info = self.env.step(a)
                if term or trunc:
                    ok = False
                    break
            if ok and self.env.unwrapped.phase >= 6:
                return obs, {}
        self.env.unwrapped.phase = 6
        return self.env.unwrapped._get_obs(), {}

    def reset(self, **kwargs):
        return self._handoff()

"""phase 6（回初始位置）训练环境：reset 时先跑 model1、model2 走到 phase 6。

交接判据固定：
  - model1（phase 1/2）用 SCHEDULE 第 5 档
  - model2（phase 3-5）用 SCHEDULE 第 6 档
"""
import os
import sys

import gymnasium as gym

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Phase6Env(gym.Wrapper):
    def __init__(self, env, model1, model2, max_handoff_steps=400, attempts=3,
                 stage1=None, stage2=None):
        super().__init__(env)
        self.model1 = model1
        self.model2 = model2
        self.max_handoff_steps = max_handoff_steps
        self.attempts = attempts
        self.stage1 = tuple(stage1) if stage1 is not None else None
        self.stage2 = tuple(stage2) if stage2 is not None else None

    def _apply(self, stage):
        if stage is None:
            return
        u = self.env.unwrapped
        u._set_dist(stage[0])
        u._set_angle1(stage[1])
        u._set_still_count(stage[2])
        if len(stage) > 3:
            u._set_rel(stage[3])

    def _reset_counters(self):
        u = self.env.unwrapped
        u.steps = 0
        u.wait_steps = 0

    def _handoff(self):
        for _ in range(self.attempts):
            obs, info = self.env.reset()
            self._apply(self.stage1)
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

            # model2 跑到 phase 6（切到第 6 档判据）
            self._apply(self.stage2)
            for _ in range(self.max_handoff_steps):
                if self.env.unwrapped.phase >= 6:
                    break
                a, _ = self.model2.predict(obs, deterministic=True)
                obs, r, term, trunc, info = self.env.step(a)
                if term or trunc:
                    ok = False
                    break

            if ok and self.env.unwrapped.phase >= 6:
                self._reset_counters()
                return obs, {}

        # ---- 兜底：强设 phase 6，并模拟"已经放料" ----
        u = self.env.unwrapped
        u.phase = 6
        self._reset_counters()
        u.attached = False          # 若 m1 已吸附，必须解除，否则方块会跟着末端跑
        if u.mocap_id >= 0:
            u.data.mocap_pos[u.mocap_id] = [float(u.drop_pos[0]),
                                            float(u.drop_pos[1]), 0.375]
            u.data.mocap_quat[u.mocap_id] = [1.0, 0.0, 0.0, 0.0]
        u.released = True
        return u._get_obs(), {}

    def reset(self, **kwargs):
        return self._handoff()

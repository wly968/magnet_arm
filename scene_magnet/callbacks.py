"""训练/诊断用的回调与工具（与 arm_env.py 解耦）。

包含：
    CurriculumCallback        —— phase 1/2 的性能门控课程（确定性探测 + 连续达标升档）
    PhaseCurriculumCallback   —— 通用阶段课程（model2/model3 用）
    PhaseTrackerCallback      —— 每回合停在哪个 phase 的计数
    DiagCallback              —— phase 2 子条件达标率（控制台）
    TrainLogCallback          —— 单行文本训练日志（saves/train_log.txt）
"""
import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    """按性能升档的课程：当前档确定性成功率达标才升到下一档。

    升档条件（同时满足）：
      1. 本档已训练 >= min_steps 步；
      2. 距上次探测 >= probe_every 步；
      3. 连续 confirm_passes 次探测的成功率都 >= success_threshold。
    探测用确定性策略在独立环境里跑 probe_episodes 局，避免采样噪声造成误判。
    """

    SCHEDULE = [
        (0.050, 0.940, 0, 0.20),
        (0.045, 0.950, 0, 0.10),
        (0.042, 0.958, 1, 0.10),
        (0.038, 0.960, 1, 0.05),
        (0.034, 0.965, 2, 0.05),
        # 终点角度 0.96≈16° 倾角，磁吸盘仍能可靠吸附（4 自由度在远端做不到 0.98）
        (0.032, 0.960, 1, 0.05),
    ]

    def __init__(self, min_steps=300_000, success_threshold=0.6,
                 target_phase=3, probe_episodes=30, probe_every=100_000,
                 confirm_passes=2, start_idx=0, verbose=1):
        super().__init__(verbose=verbose)
        self.min_steps = int(min_steps)
        self.success_threshold = float(success_threshold)
        self.target_phase = int(target_phase)
        self.probe_episodes = int(probe_episodes)
        self.probe_every = int(probe_every)
        self.confirm_passes = int(confirm_passes)
        n_stages = len(self._stages())
        self.idx = max(0, min(int(start_idx), n_stages - 1))
        self.stage_start = 0
        self.ep_ok = 0
        self.ep_total = 0
        self.last_probe_rate = float("nan")
        self._started = False
        self._probe_env = None
        self._last_probe = -10**18
        self._pass_streak = 0
        self.still_target = "still"   # "still"=phase 2/5 的 still；"home_still"=phase 6 的 dwell

    def _stages(self):
        out = []
        for t in self.SCHEDULE:
            if len(t) == 5:
                out.append(tuple(t[1:]))
            elif len(t) == 3:
                out.append(tuple(t) + (0.0,))
            else:
                out.append(tuple(t))
        return out

    def _apply(self, stage):
        dist, angle, still, rel = stage
        self.training_env.env_method("_set_dist", dist)
        self.training_env.env_method("_set_angle1", angle)
        if self.still_target == "home_still":
            self.training_env.env_method("_set_home_still", still)
        else:
            self.training_env.env_method("_set_still_count", still)
        self.training_env.env_method("_set_rel", rel)
        if self.verbose:
            print(f"[curriculum] 第 {self.idx + 1}/{len(self._stages())} 档 | "
                  f"dist {dist} angle {angle} still {still} rel {rel}")

    def _probe(self, stage):
        if self._probe_env is None:
            from envs.arm_env import ArmEnv      # 惰性导入，避免循环依赖
            self._probe_env = ArmEnv()
        e = self._probe_env
        dist, angle, still, rel = stage
        e._set_dist(dist)
        e._set_angle1(angle)
        e._set_still_count(still)
        e._set_rel(rel)
        ok = 0
        for _ in range(self.probe_episodes):
            obs, _ = e.reset()
            for _ in range(e.max_steps + e.max_wait_steps + 10):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = e.step(action)
                if e.phase >= self.target_phase or term or trunc:
                    break
            ok += int(e.phase >= self.target_phase)
        return ok / max(1, self.probe_episodes)

    def _on_step(self) -> bool:
        if not self._started:
            self._apply(self._stages()[self.idx])
            self.stage_start = self.num_timesteps
            self._started = True

        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", None)
        if dones is None:
            dones = [False] * len(infos)
        for info, done in zip(infos, dones):
            if done:
                self.ep_total += 1
                if info.get("final_phase", 1) >= self.target_phase:
                    self.ep_ok += 1

        stages = self._stages()
        elapsed = self.num_timesteps - self.stage_start
        # 最后一档不再升档，但仍然要探测（否则日志里的 probe 会一直停在旧值）
        if (elapsed >= self.min_steps
                and self.num_timesteps - self._last_probe >= self.probe_every):
            self._last_probe = self.num_timesteps
            rate = self._probe(stages[self.idx])
            self.last_probe_rate = rate
            ok = rate >= self.success_threshold
            self._pass_streak = self._pass_streak + 1 if ok else 0
            if self.verbose:
                print(f"[curriculum] 第 {self.idx + 1} 档探测：确定性成功率 {rate:.0%}"
                      f"（阈值 {self.success_threshold:.0%}，连续达标 "
                      f"{self._pass_streak}/{self.confirm_passes}）")
            if self.idx + 1 >= len(stages):
                self._pass_streak = 0
            elif self._pass_streak >= self.confirm_passes:
                self.idx += 1
                self._apply(stages[self.idx])
                self.stage_start = self.num_timesteps
                self.ep_ok = 0
                self.ep_total = 0
                self._pass_streak = 0
                if self.verbose:
                    print(f"[curriculum] 升档成功（连续 {self.confirm_passes} 次达标）")
        return True

class PhaseCurriculumCallback(BaseCallback):
    """与 CurriculumCallback 同款门控，但用于 model2/model3 的阶段阈值。

    成功 = 回合内到达 target_phase。每档用确定性探测 N 局，
    连续 confirm_passes 次达标才升档。
    stages: [(dist, angle, still) 或 (dist, angle, still, home_tol), ...]
    """

    def __init__(self, probe_env_fn, stages, target_phase=6, min_steps=200_000,
                 success_threshold=0.6, probe_episodes=20, probe_every=100_000,
                 confirm_passes=2, start_idx=0, home_tol=None,
                 still_target="still", verbose=1):
        super().__init__(verbose=verbose)
        self.probe_env_fn = probe_env_fn
        self.stages = [tuple(s) for s in stages]
        self.target_phase = int(target_phase)
        self.min_steps = int(min_steps)
        self.success_threshold = float(success_threshold)
        self.probe_episodes = int(probe_episodes)
        self.probe_every = int(probe_every)
        self.confirm_passes = int(confirm_passes)
        self.home_tol = home_tol
        self.still_target = still_target
        self.idx = max(0, min(int(start_idx), len(self.stages) - 1))
        self.stage_start = 0
        self.last_probe_rate = float("nan")
        self._started = False
        self._probe_env = None
        self._last_probe = -10**18
        self._pass_streak = 0

    def _apply(self, stage):
        dist, angle, still = stage[0], stage[1], stage[2]
        rel = stage[3] if len(stage) > 3 else None
        self.training_env.env_method("_set_dist", dist)
        self.training_env.env_method("_set_angle1", angle)
        if self.still_target == "home_still":
            self.training_env.env_method("_set_home_still", still)
        else:
            self.training_env.env_method("_set_still_count", still)
        if rel is not None:
            self.training_env.env_method("_set_rel", rel)
        if self.home_tol is not None:
            self.training_env.env_method("_set_home_tol", self.home_tol)
        if self.verbose:
            extra = f" rel {rel}" if rel is not None else ""
            print(f"[phase-curriculum] 第 {self.idx + 1}/{len(self.stages)} 档 | "
                  f"dist {dist} angle {angle} still {still}{extra}")

    def _probe(self):
        if self._probe_env is None:
            self._probe_env = self.probe_env_fn()
        e = self._probe_env
        ok = 0
        for _ in range(self.probe_episodes):
            obs, _ = e.reset()
            for _ in range(500):
                a, _ = self.model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = e.step(a)
                if term or trunc:
                    break
            ok += int(e.unwrapped.phase >= self.target_phase)
        return ok / max(1, self.probe_episodes)

    def _on_step(self) -> bool:
        if not self._started:
            self._apply(self.stages[self.idx])
            self.stage_start = self.num_timesteps
            self._started = True

        elapsed = self.num_timesteps - self.stage_start
        if (elapsed >= self.min_steps
                and self.num_timesteps - self._last_probe >= self.probe_every):
            self._last_probe = self.num_timesteps
            rate = self._probe()
            self.last_probe_rate = rate
            self._pass_streak = self._pass_streak + 1 if rate >= self.success_threshold else 0
            if self.verbose:
                print(f"[phase-curriculum] 第 {self.idx + 1} 档探测：成功率 {rate:.0%}"
                      f"（阈值 {self.success_threshold:.0%}，连续 {self._pass_streak}/"
                      f"{self.confirm_passes}）")
            if self.idx + 1 < len(self.stages) and self._pass_streak >= self.confirm_passes:
                self.idx += 1
                self._apply(self.stages[self.idx])
                self.stage_start = self.num_timesteps
                self._pass_streak = 0
                if self.verbose:
                    print("[phase-curriculum] 升档成功")
        return True

class PhaseTrackerCallback(BaseCallback):
    """统计每回合结束在哪个 phase（P1~P6，7=完整成功）。"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.counts = {i: 0 for i in range(1, 8)}

    def _on_step(self) -> bool:
        for info, done in zip(self.locals.get("infos", []), self.locals.get("dones", [])):
            if done:
                fp = int(info.get("final_phase", 0))
                if fp in self.counts:
                    self.counts[fp] += 1
        if self.n_calls % (self.model.n_steps * 10) == 0:
            total = sum(self.counts.values())
            if total > 0:
                parts = " / ".join(f"P{k} {v}" for k, v in self.counts.items() if v)
                print(f"\n阶段统计（{total} 局）：{parts}\n")
                self.counts = {i: 0 for i in range(1, 8)}
        return True

class DiagCallback(BaseCallback):
    """定期汇总 phase 2 各子条件的满足比例，定位卡在哪一项。"""

    def __init__(self, every=20_000, verbose=0):
        super().__init__(verbose=verbose)
        self.every = int(every)
        self._freq = 1

    def _on_training_start(self):
        n = max(1, self.training_env.num_envs)
        self._freq = max(1, self.every // n)

    def _on_step(self) -> bool:
        if self.n_calls % self._freq != 0:
            return True
        diags = self.training_env.env_method("_diag")
        n = sum(d["n"] for d in diags)
        if n == 0:
            return True

        def pct(key):
            return 100.0 * sum(d[key] for d in diags) / n

        md = sum(d["sum_dist"] for d in diags) / n
        ma = sum(d["sum_angle"] for d in diags) / n
        mz = sum(d["sum_dz"] for d in diags) / n
        mr = sum(d["sum_rel"] for d in diags) / n
        n_pos = sum(d["n_pos"] for d in diags)
        rel_pos = sum(d["rel_pos"] for d in diags)
        mrp = (sum(d["sum_rel_pos"] for d in diags) / n_pos) if n_pos else float("nan")
        p_pos = 100.0 * n_pos / n
        p_rel_pos = (100.0 * rel_pos / n_pos) if n_pos else float("nan")
        try:
            thr = self.training_env.get_attr("dist_threshold")[0]
            tha = self.training_env.get_attr("angle1_threshold")[0]
            thr_rel = self.training_env.get_attr("rel_threshold")[0]
        except Exception:
            thr = tha = thr_rel = float("nan")
        print(f"[diag {self.num_timesteps}] 阈 {thr:.3f}/{tha:.2f}/rel {thr_rel:.2f} | "
              f"帧 {n:6d} | 位置到位 {p_pos:5.1f}% (n={n_pos:5d}) | "
              f"到位中 rel 达标 {p_rel_pos:5.1f}% | 全满足 {pct('all'):5.1f}% | "
              f"到位时均值 rel {mrp:.3f} | 总体均值 rel {mr:.3f} | "
              f"dist {md:.3f} angle {ma:.3f} dz {mz:+.3f}")
        return True

class TrainLogCallback(BaseCallback):
    """每 every 步写一行纯文本汇总；课程升档时额外写一行 STAGE 标记。"""

    def __init__(self, log_path, every=20_000, curriculum=None, verbose=1):
        super().__init__(verbose=verbose)
        self.log_path = log_path
        self.every = int(every)
        self.curriculum = curriculum
        self._freq = 1
        self._last_stage = None
        self._ends = {"succ": 0, "tout": 0, "nan": 0}

    def _write(self, line):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if self.verbose:
            print(line)

    def _on_training_start(self):
        n = max(1, self.training_env.num_envs)
        self._freq = max(1, self.every // n)
        d = os.path.dirname(self.log_path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("# step stage thr(dist/angle/still/rel) probe pos relpos all "
                    "mrel mdist mang mdz succ tout nan std ev kl\n")
        self._last_stage = None
        self._ends = {"succ": 0, "tout": 0, "nan": 0}

    def _on_step(self) -> bool:
        # 0) 结束原因统计（直接来自 env 的 info，不经过 ep_info_buffer）
        for info, done in zip(self.locals.get("infos", []),
                              self.locals.get("dones", [False] * 0)):
            if not done:
                continue
            if info.get("final_phase", 1) >= 3:
                self._ends["succ"] += 1
            elif info.get("nan"):
                self._ends["nan"] += 1
            else:
                self._ends["tout"] += 1

        # 1) 升档标记（每步都检查，开销极小）
        if self.curriculum is not None:
            cur = self.curriculum.idx + 1
            if self._last_stage is None:
                self._last_stage = cur
            elif cur != self._last_stage:
                probe = self.curriculum.last_probe_rate
                thr = self.curriculum._stages()[self.curriculum.idx]
                self._write(f"{self.num_timesteps} STAGE {self._last_stage}->{cur} "
                            f"probe={probe:.0%} "
                            f"thr={thr[0]:.3f}/{thr[1]:.2f}/{int(thr[2])}/{thr[3]:.2f}")
                self._last_stage = cur

        # 2) 周期汇总
        if self.n_calls % self._freq != 0:
            return True

        diags = self.training_env.env_method("_diag")
        n = sum(d["n"] for d in diags)
        n_pos = sum(d["n_pos"] for d in diags)
        rel_pos = sum(d["rel_pos"] for d in diags)

        def mean(key, denom):
            return (sum(d[key] for d in diags) / denom) if denom else float("nan")

        try:
            thr = self.training_env.get_attr("dist_threshold")[0]
            tha = self.training_env.get_attr("angle1_threshold")[0]
            ths = self.training_env.get_attr("still_count_threshold")[0]
            thr_rel = self.training_env.get_attr("rel_threshold")[0]
        except Exception:
            thr = tha = ths = thr_rel = float("nan")

        ep_len = ep_rew = float("nan")
        if getattr(self.model, "ep_info_buffer", None):
            buf = list(self.model.ep_info_buffer)
            if buf:
                ep_len = float(np.mean([e["l"] for e in buf]))
                ep_rew = float(np.mean([e["r"] for e in buf]))

        lv = getattr(self.logger, "name_to_value", {}) or {}
        std = float(np.exp(self.model.policy.log_std.mean().item()))
        stage = (self.curriculum.idx + 1) if self.curriculum is not None else 0
        probe = (self.curriculum.last_probe_rate
                 if self.curriculum is not None else float("nan"))

        tot_end = sum(self._ends.values())
        line = (
            f"{self.num_timesteps} s{stage} "
            f"thr={thr:.3f}/{tha:.2f}/{int(ths)}/{thr_rel:.2f} "
            f"probe={probe:.0%} "
            + (f"pos={100.0*n_pos/n:.1f}% relpos={100.0*rel_pos/n_pos:.1f}% "
               f"all={100.0*sum(d['all'] for d in diags)/n:.1f}% "
               f"mrel={mean('sum_rel', n):.3f} mdist={mean('sum_dist', n):.3f} "
               f"mang={mean('sum_angle', n):.3f} mdz={mean('sum_dz', n):+.3f} "
               if n else "pos=- relpos=- all=- mrel=- mdist=- mang=- mdz=- ")
            + f"succ={100.0*self._ends['succ']/max(1, tot_end):.0f}% "
              f"tout={100.0*self._ends['tout']/max(1, tot_end):.0f}% "
              f"nan={100.0*self._ends['nan']/max(1, tot_end):.0f}% "
              f"std={std:.3f} "
              f"ev={float(lv.get('train/explained_variance', float('nan'))):.2f} "
              f"kl={float(lv.get('train/approx_kl', float('nan'))):.4f}"
        )
        self._write(line)
        self.training_env.env_method("_reset_diag")   # 由本回调负责清零
        self._ends = {"succ": 0, "tout": 0, "nan": 0}
        return True

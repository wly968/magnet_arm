import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

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


class WaitAwareMonitor(Monitor):
    """训练日志里的 ep_len 去掉 phase 0 的等待步数。"""

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        episode = info.get("episode")
        if episode is not None:
            episode["l"] = max(0, int(episode["l"]) - int(info.get("wait_steps", 0)))
        return obs, reward, terminated, truncated, info


class ArmEnv(gym.Env):
    """phase 1/2 机械臂任务。

    phase 0: 等待方块沿带走到取料位置（手臂保持初始位）
    phase 1: 到方块上方 15cm
    phase 2: 下压到方块上方约 6cm，满足 dist/angle 即成功（抓取）

    各阶段是否结束回合，直接改对应分支 return 里的第三个布尔值即可：
      return self._get_obs(), reward, True/False, truncated, info
      True = 该阶段完成后结束回合并算成功；False = 继续下一阶段
    """

    def __init__(self):
        super().__init__()
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        xml_path = os.path.join(base_dir, "assets", "models", "train_scene.xml")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.home_ctrl = np.array([1.57, -1.15, 2.0, 0.725], dtype=np.float64)
        self.ctrl_scale = np.array([3.0, 1.0, 1.0, 1.0], dtype=np.float64)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32
        )

        # 目标点
        self.pick_offset = np.array([0.0, 0.0, 0.15])
        self.grasp_offset = np.array([0.0, 0.0, 0.06])
        self.transfer_pose = np.array([0.42, 0.27, 0.78])
        # 试过 [0.25, 0.25, 0.78]（yaw 行程更均衡，但 model2 需重训，否则 24/30 卡在 phase 4）
        self.drop_pos = np.array([0.65, -0.45, 0.435])
        self.attach_offset = np.array([0.0, 0.0, -0.3])   # 未来磁吸用

        # 判据
        self.dist_threshold = 0.05
        self.angle1_threshold = 0.94
        self.still_count_threshold = 0
        self.still_count = 0
        self.diag = self._empty_diag()
        self.vel_threshold = 0.15
        self.home_tol = 0.15   # phase 6 判定：关节角与 home 的偏差
        self.home_still_threshold = 2   # phase 6：q_err 需连续处于容差内的帧数（>N 帧）

        self.cube_hit = False
        # 抓后抬起：由环境确定性执行，训练/测试一致，避免策略在交接处“勾一下”
        self.scripted_retract = True
        self.retract_steps = 0

        self.steps = 0
        self.max_steps = 400
        # phase 1 只是"接近"，用固定宽松阈值，不随课程收紧（课程只作用于 phase 2）
        self.phase1_dist = 0.050
        self.phase1_angle = 0.940
        # phase 4 只是"搬运经过点"，不要求工具竖直（竖直解在该点需要大幅换构型）
        self.phase4_angle = 0.600
        self.phase = 1

        self.arm_joints = [
            self.model.joint(n).id
            for n in ("joint_yaw", "joint_shoulder", "joint_elbow", "joint_wrist")
        ]
        self.arm_qpos_adr = self.model.jnt_qposadr[self.arm_joints]
        self.arm_dof_adr = self.model.jnt_dofadr[self.arm_joints]

        self.box_body_id = self.model.body("cube_0").id
        self.box_geom_id = self.model.geom("cube_0_geom").id
        self.mocap_id = int(self.model.body_mocapid[self.box_body_id])   # 方块是 mocap
        self.box_pos = np.zeros(3)
        self.attached = False       # phase2 后锁定到末端，phase5 后解锁
        self.carry_offset = 0.06
        self.smooth_alpha = 0.0      # >0 时对动作做低通滤波，抑制“勾一下”
        # IK 基准 + RL 残差（phase 1/2）
        self.ik_phases = (1, 2)
        self.ik_scale = 0.02         # 残差尺度 rad/单位动作（末端约 5 mm/步）
        self.ik_iters = 5
        self.ik_lr = 1.0
        self.ik_damping = 1e-3
        # 视觉不确定性占位（接入视觉前恒为 0）：σx, σy, σvx, σvy
        self.pos_std = np.zeros(4, dtype=np.float64)

        # 传送带运动 + phase 0 等待
        self.cube_speed = 0.20                          # 带 A 输送速度 (m/s)，0 = 静止
        self.belt_dir = np.array([1.0, 0.0, 0.0])        # 带 A 沿 +x
        self.spawn_mode = "gauss"                      # "uniform" | "gauss" | "fixed"
        self.spawn_x_range = (-0.700, -0.600)            # x：靠近抓取线，等待约 60~110 步
        self.spawn_x_mean = -0.650                       # 高斯分布均值（= 范围中心）
        self.spawn_x_std = 0.025                         # 高斯分布标准差（范围 ±0.05，两端仅轻微裁剪）
        self.spawn_x = -1.150                            # fixed 模式下的固定值
        self.spawn_y_range = (0.530, 0.770)              # y：关于带面中心 0.65 对称，半宽 0.12
        self.spawn_y = 0.650                             # fixed 模式下的固定 y
        self.arrive_x = -0.475                           # 原刷新区最左端：到达后进入 phase 1
        self.max_wait_steps = 900                        # phase 0 等待上限（不计入回合预算）
        self.wait_steps = 0
        self.box_vel = np.zeros(3)                       # 方块实际速度（观测用）
        self.phase0_hold_tol = 0.06                      # phase 0：关节偏差小于此值时保持静止
        # B 带输送 + 收集篮
        self.beltB_dir = np.array([0.0, -1.0, 0.0])
        self.beltB_speed = 0.8                           # 放料后方块沿 B 带滑向篮子
        self.beltB_end_y = -1.18                         # 超过此位置视为进入篮子
        self.basket_pos = np.array([0.65, -1.34, 0.375])
        self.collect_after_place = True
        self.released = False
        self.collected = False
        self.prev_cmd = np.zeros(4, dtype=np.float64)
        self.phase_entry_steps = 0   # 阶段切换后的强平滑步数

        self.arm_geom_ids = set()
        for _bn in ("turret", "shoulder", "elbow", "wrist", "gripper"):
            _bid = self.model.body(_bn).id
            for _g in range(self.model.body_geomadr[_bid],
                            self.model.body_geomadr[_bid] + self.model.body_geomnum[_bid]):
                self.arm_geom_ids.add(int(_g))

        self._jp_cache = np.zeros((3, self.model.nv))
        self.jnt_lo = self.model.jnt_range[self.arm_joints][:, 0].copy()
        self.jnt_hi = self.model.jnt_range[self.arm_joints][:, 1].copy()
        self.ee_id = self.model.site("ee_site").id
        self.ee_val1 = np.zeros(6)
        self.rel_threshold = 0.03

        self.prev_action = np.zeros(4, dtype=np.float64)
        self.prev_action2 = np.zeros(4, dtype=np.float64)
        self.phase2_eepos = self.box_pos[2] + 0.04

    # ---------------- 基本 ----------------

    def _ik_target_q(self, target):
        """对目标点做位置 IK，返回 4 个关节角（不改变仿真状态）。"""
        q_save = self.data.qpos[self.arm_qpos_adr].copy()
        q = q_save.copy()
        jp = self._jp_cache if getattr(self, "_jp_cache", None) is not None else np.zeros((3, self.model.nv))
        for _ in range(self.ik_iters):
            self.data.qpos[self.arm_qpos_adr] = q
            mujoco.mj_kinematics(self.model, self.data)
            err = np.asarray(target) - self.data.site_xpos[self.ee_id]
            jp[:] = 0.0
            mujoco.mj_jacSite(self.model, self.data, jp, None, self.ee_id)
            J = jp[:, self.arm_dof_adr]
            # 梯度迭代（不用 np.linalg.solve：本机 LAPACK 会硬崩 0xc06d007f）
            dq = J.T @ err
            nrm = float(np.linalg.norm(dq))
            if nrm > 1e-9:
                dq *= min(1.0, 0.35 / nrm) * 1.0
            q = np.clip(q + self.ik_lr * dq, self.jnt_lo, self.jnt_hi)
        self.data.qpos[self.arm_qpos_adr] = q_save
        mujoco.mj_forward(self.model, self.data)
        return q

    def _retract_action(self, target_z):
        """确定性抬升：保持当前关节，用肩关节闭环把末端抬到 target_z。"""
        q = self.data.qpos[self.arm_qpos_adr]
        a = (q - self.home_ctrl) / self.ctrl_scale
        err = target_z - float(self.data.site_xpos[self.ee_id][2])
        a[1] += -2.5 * err
        return np.clip(a, -1.0, 1.0)

    def _target(self):
        if self.phase == 0:            # 等待：保持在初始位
            return self.home_ee
        if self.phase == 1:            # 取料点（悬停）
            return self.box_pos + self.pick_offset
        if self.phase == 2:            # 下抓
            return self.box_pos + self.grasp_offset
        if self.phase == 3:            # 抓后竖直抬起，回到悬停高度
            return self.box_pos + self.pick_offset
        if self.phase == 4:            # 中间转移姿态
            return self.transfer_pose
        if self.phase == 5:            # 放料
            return self.drop_pos
        return self.home_ee            # 回到初始位置

    def reset(self, seed=None, option=None):
        super().reset(seed=seed)

        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        # 生成点：在线随机 / 高斯 / 固定
        if self.spawn_mode == "fixed":
            spawn_x = self.spawn_x
        elif self.spawn_mode == "gauss":
            lo, hi = self.spawn_x_range
            spawn_x = float(np.clip(self.np_random.normal(self.spawn_x_mean, self.spawn_x_std), lo, hi))
        else:  # uniform
            spawn_x = float(self.np_random.uniform(*self.spawn_x_range))
        spawn_y = float(self.spawn_y) if self.spawn_mode == "fixed" else float(self.np_random.uniform(*self.spawn_y_range))
        self.box_pos = np.array([spawn_x, spawn_y, 0.375])
        self.attached = False
        self.released = False
        self.collected = False
        if self.mocap_id >= 0:
            self.data.mocap_pos[self.mocap_id] = self.box_pos
            self.data.mocap_quat[self.mocap_id] = [1.0, 0.0, 0.0, 0.0]   # 保持竖直
        mujoco.mj_forward(self.model, self.data)

        self.box_vel = self.belt_dir * self.cube_speed
        self.home_ee = self.data.site_xpos[self.ee_id].copy()
        self.home_qpos = self.data.qpos[self.arm_qpos_adr].copy()
        self.steps = 0
        self.wait_steps = 0
        self.phase = 0            # phase 0：方块还没到，手臂在初始位等待
        self.still_count = 0
        self.prev_cmd = np.zeros(4, dtype=np.float64)
        self.retract_steps = 0
        self.prev_action = np.zeros(4, dtype=np.float64)
        self.prev_action2 = np.zeros(4, dtype=np.float64)
        self.prev_action2 = np.zeros(4, dtype=np.float64)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        # 低通滤波：a_cmd = alpha*a_prev_cmd + (1-alpha)*a
        # 阶段刚切换时用更强的 alpha，压掉交接处的“勾一下”
        alpha = self.smooth_alpha
        if self.phase_entry_steps > 0:
            alpha = max(alpha, 0.85)
            self.phase_entry_steps -= 1
        if alpha > 0.0:
            action = alpha * self.prev_cmd + (1.0 - alpha) * action

        # phase 0：已经接近初始姿态时锁住，避免原地抽搐
        if self.phase == 0 and self.phase0_hold_tol > 0.0:
            q_err_hold = float(np.linalg.norm(self.data.qpos[self.arm_qpos_adr] - self.home_qpos))
            if q_err_hold < self.phase0_hold_tol:
                # 固定住 home 目标（不要跟随当前姿态，否则会被重力一点点带下去）
                action = (self.home_qpos - self.home_ctrl) / self.ctrl_scale
        self.prev_cmd = action.copy()

        if self.scripted_retract and self.phase == 3:
            action = self._retract_action(float(self.box_pos[2]) + 0.15)
            self.retract_steps += 1
            if self.retract_steps > 60:          # 卡住时兜底，避免死循环
                self.phase = 4

        prev_box = self.box_pos.copy()

        dt = self.model.opt.timestep * 5
        if self.released and not self.collected and self.mocap_id >= 0:
            # 放到 B 带后：沿 -y 滑向收集篮
            mp = self.data.mocap_pos[self.mocap_id]
            mp[:] = mp + self.beltB_dir * self.beltB_speed * dt
            if mp[1] <= self.beltB_end_y:
                self.data.mocap_pos[self.mocap_id] = self.basket_pos
                self.collected = True
        elif not self.attached and self.cube_speed > 0.0 and self.mocap_id >= 0:
            # 未释放：方块沿带 A 匀速前进（运动学）
            mp = self.data.mocap_pos[self.mocap_id]
            mp[:] = mp + self.belt_dir * self.cube_speed * dt

        # 锁定期间：方块跟随末端，姿态保持竖直
        if self.attached and self.mocap_id >= 0:
            ee_now = self.data.site_xpos[self.ee_id]
            self.data.mocap_pos[self.mocap_id] = [float(ee_now[0]),float(ee_now[1]),float(ee_now[2]) - self.carry_offset]
            self.data.mocap_quat[self.mocap_id] = [1.0, 0.0, 0.0, 0.0]

        if self.phase in self.ik_phases:
            q_ref = self._ik_target_q(self._target())
            self.data.ctrl[:] = q_ref + action * self.ik_scale
        else:
            self.data.ctrl[:] = self.home_ctrl + action * self.ctrl_scale
        mujoco.mj_step(self.model, self.data, 5)

        if self.phase == 0:
            self.wait_steps += 1
            truncated = self.wait_steps >= self.max_wait_steps
        else:
            self.steps += 1
            truncated = self.steps >= self.max_steps

        if np.isnan(self.data.qacc).any():
            return self._get_obs(), -10.0, True, False, {"nan": True, "final_phase": self.phase}

        info = {"final_phase": self.phase, "wait_steps": self.wait_steps,
                "collected": bool(self.collected)}
        self.box_pos = self.data.xpos[self.box_body_id].copy()
        if self.mocap_id >= 0:
            self.box_pos = self.data.mocap_pos[self.mocap_id].copy()
        dt_step = self.model.opt.timestep * 5
        self.box_vel = (self.box_pos - prev_box) / max(dt_step, 1e-9)
        ee_pos = self.data.site_xpos[self.ee_id].copy()
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_SITE,self.ee_id,self.ee_val1,0)
        self.ee_val = self.ee_val1[3:]
        target_pos = self._target()
        dist = float(min(np.linalg.norm(ee_pos - target_pos), 1.0))

        joint_3_pos = self.data.xanchor[self.arm_joints[3]].copy()
        forward = ee_pos - joint_3_pos
        forward = forward / np.linalg.norm(forward)
        angle1 = float(np.dot(forward, np.array([0, 0, -1])))
        action_change = float(np.sum(np.square(action - self.prev_action)))
        jerk = float(np.sum(np.square(action - 2.0 * self.prev_action + self.prev_action2)))
        self.prev_action2 = self.prev_action.copy()
        self.prev_action = action.copy()
        rel = np.linalg.norm(self.ee_val - self.box_vel)

        self.cube_hit = False
        # 抓后抬起：由环境确定性执行，训练/测试一致，避免策略在交接处“勾一下”
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == self.box_geom_id and c.geom2 in self.arm_geom_ids:
                self.cube_hit = True
                break
            if c.geom2 == self.box_geom_id and c.geom1 in self.arm_geom_ids:
                self.cube_hit = True
                break

        if self.phase == 0:
            # 等待阶段：与 phase 6 同款“回到初始姿态”奖励
            q_err = float(np.linalg.norm(self.data.qpos[self.arm_qpos_adr] - self.home_qpos))
            reward = -q_err * 2.0 - action_change * 0.3
            if float(self.box_pos[0]) >= self.arrive_x:      # 方块到达原刷新点最左端
                reward += 10.0
                self.phase = 1
                info["final_phase"] = 1
                return self._get_obs(), reward, False, truncated, info

        elif self.phase == 1:
            reward = -dist * 4.0 - (1.0 - angle1) ** 2 * 3.0 - action_change * 0.3
            if dist < self.phase1_dist and angle1 > self.phase1_angle:
                reward += 30.0
                self.phase = 2
                info["final_phase"] = 2
                # ↓ 第三个值：True = 到此结束回合，False = 继续 phase 2
                return self._get_obs(), reward, False, False, info

        elif self.phase == 2:
            reward = -dist * 4.0 -  5.0 * (1.0 - angle1) - action_change * 0.3 - 0.2 + 0.3*min(self.still_count,5)
            # ---- 诊断统计：分别记录各子条件的满足情况 ----
            _d_ok = dist < self.dist_threshold
            _a_ok = angle1 > self.angle1_threshold
            _r_ok = float(rel) < self.rel_threshold
            _b_ok = (ee_pos[2] < target_pos[2] + 0.03) and (ee_pos[2] > float(self.box_pos[2]) + 0.03)
            _g = self.diag
            _g["n"] += 1
            _g["dist"] += int(_d_ok)
            _g["angle"] += int(_a_ok)
            _g["rel"] += int(_r_ok)
            _g["band"] += int(_b_ok)
            _g["all"] += int(_d_ok and _a_ok and _r_ok and _b_ok)
            _g["sum_dist"] += dist
            _g["sum_angle"] += angle1
            _g["sum_dz"] += float(ee_pos[2]) - float(target_pos[2])
            _g["sum_rel"] += float(rel)
            _pos = _d_ok and _a_ok and _b_ok          # 其余三项都到位
            _g["n_pos"] += int(_pos)
            _g["rel_pos"] += int(_pos and _r_ok)
            if _pos:
                _g["sum_rel_pos"] += float(rel)
            # 允许略压过目标平面，但末端不得低于方块顶面 5cm（避免圆盘嵌进方块）
            if dist < self.dist_threshold and angle1 > self.angle1_threshold and np.linalg.norm(self.ee_val-self.box_vel)<self.rel_threshold and ee_pos[2] < target_pos[2] + 0.03 and ee_pos[2] > self.box_pos[2] + 0.03:
                self.still_count += 1
            if not (dist < self.dist_threshold and angle1 > self.angle1_threshold and np.linalg.norm(self.ee_val-self.box_vel)<self.rel_threshold and ee_pos[2] < target_pos[2] + 0.03 and ee_pos[2] > self.box_pos[2] + 0.03):
                self.still_count = 0
            if dist < self.dist_threshold and angle1 > self.angle1_threshold and ee_pos[2] < target_pos[2] + 0.03 and ee_pos[2] > self.box_pos[2] + 0.03 and self.still_count > self.still_count_threshold :
                reward += 30.0
                self.phase = 3
                self.still_count = 0
                self.attached = True              # 抓住：方块跟随末端
                info["final_phase"] = 3
                # ↓ 第三个值：True = 到此结束回合，False = 继续 phase 3
                return self._get_obs(), reward, False, truncated, info

        elif self.phase == 3:
            # 抓后竖直抬起（动作由 _retract_action 确定性给出）
            reward = -dist * 4.0 - (1.0 - angle1) ** 2 * 3.0 - action_change * 0.3
            ee_z = float(ee_pos[2])
            if self.retract_steps >= 18 or (ee_z > float(self.box_pos[2]) + 0.12 and self.retract_steps >= 4):
                reward += 30.0
                self.phase = 4
                info["final_phase"] = 4
                return self._get_obs(), reward, False, truncated, info

        elif self.phase == 4:
            # 转移段：加大动作变化与 jerk 惩罚，消除“勾一下”
            reward = -dist * 4.0 - 5.0 * (1.0 - angle1) - action_change * 0.3
            if dist < self.dist_threshold and angle1 > self.phase4_angle:
                reward += 30.0
                self.phase = 5
                info["final_phase"] = 5
                return self._get_obs(), reward, False, truncated, info

        elif self.phase == 5:
            # 放料：到位且静止（避免把方块甩出去）才算完成
            reward = -dist * 4.0 - (1.0 - angle1) ** 2 * 3.0 - action_change * 0.3 + 0.3*min(self.still_count,5)
            qvel_norm = float(np.linalg.norm(self.data.qvel[self.arm_dof_adr]))
            in_pos = dist < self.dist_threshold and angle1 > self.angle1_threshold
            if in_pos and qvel_norm < self.vel_threshold:
                self.still_count += 1
            else:
                self.still_count = 0
            if in_pos:
                reward += 0.5 * (1.0 - min(1.0, qvel_norm / max(self.vel_threshold, 1e-6)))
            if in_pos and self.still_count >= self.still_count_threshold:
                reward += 60.0
                self.still_count = 0
                self.phase = 6
                self.attached = False        # 放下：方块解锁
                self.released = True         # 交给 B 带，滑向收集篮
                info["final_phase"] = 6
                # ↓ 第三个值：True = 放料后结束（只训到 phase 5 时用），False = 继续回原位
                return self._get_obs(), reward, False, truncated, info

        else:
            # phase 6：回到初始位置；若开启收集，需等方块进入篮子
            q_err = float(np.linalg.norm(self.data.qpos[self.arm_qpos_adr] - self.home_qpos))
            reward = -q_err * 2.0 - action_change * 0.3
            # 连续多帧都在容差内才算"停稳"（噪声骗不过）
            if q_err < self.home_tol:
                self.still_count += 1
            else:
                self.still_count = 0
            reward += 0.3 * min(self.still_count, 5)
            if q_err < self.home_tol and self.still_count > self.home_still_threshold:
                if self.collected or not self.collect_after_place:
                    reward += 60.0
                    self.phase = 7
                    self.still_count = 0
                    info["final_phase"] = 7
                    info["success"] = True
                    info["collected"] = bool(self.collected)
                    return self._get_obs(), reward, True, False, info
                reward += 0.5          # 保持初始位，等方块入篮

        return self._get_obs(), reward, False, truncated, info


    def _get_obs(self):
        qpos = self.data.qpos[self.arm_qpos_adr].copy()
        qvel = self.data.qvel[self.arm_dof_adr].copy()
        ee_pos = self.data.site_xpos[self.ee_id].copy()
        target_pos = self._target().copy()
        rel_pos = target_pos - ee_pos
        phase = np.array([float(self.phase)])
        cube_speed = np.array([float(self.box_vel[0])])   # 方块沿带速度
        # 4+4+3+3+3+1+1 = 19，再加视觉不确定性 4 维（暂为 0）= 23
        return np.concatenate(
            [qpos, qvel, ee_pos, target_pos, rel_pos, phase, cube_speed, self.pos_std]
        ).astype(np.float32)

    # ---------------- 课程接口 ----------------

    @staticmethod
    def _empty_diag():
        return {"n": 0, "dist": 0, "angle": 0, "rel": 0, "band": 0, "all": 0,
                "n_pos": 0, "rel_pos": 0,
                "sum_dist": 0.0, "sum_angle": 0.0, "sum_dz": 0.0,
                "sum_rel": 0.0, "sum_rel_pos": 0.0}

    def _diag(self):
        return dict(self.diag)

    def _reset_diag(self):
        self.diag = self._empty_diag()

    def _set_dist(self, dist):
        self.dist_threshold = dist

    def _set_angle1(self, angle):
        self.angle1_threshold = angle

    def _set_still_count(self, still_count):
        self.still_count_threshold = still_count

    def _set_rel(self,rel):
        self.rel_threshold = rel

    def _set_home_tol(self, value):
        self.home_tol = float(value)

    def _set_home_still(self, value):
        self.home_still_threshold = int(value)
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    """平滑课程：每档只微调一项。"""

    SCHEDULE = [
        (0,         0.050, 0.940, 0),
        (250_000,   0.045, 0.950, 0),
        (500_000,   0.042, 0.958, 1),
        (800_000,   0.038, 0.965, 1),
        (1_100_000, 0.034, 0.972, 2),
        (1_400_000, 0.030, 0.980, 2),
    ]

    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)
        self._stage = None

    def _on_step(self) -> bool:
        stage = self.SCHEDULE[0]
        for cand in self.SCHEDULE:
            if self.num_timesteps >= cand[0]:
                stage = cand
        if stage != self._stage:
            _, dist, angle, still = stage
            self.training_env.env_method("_set_dist", dist)
            self.training_env.env_method("_set_angle1", angle)
            self.training_env.env_method("_set_still_count", still)
            self._stage = stage
            if self.verbose:
                print(f"[curriculum] {self.num_timesteps} -> dist {dist} angle {angle} still {still}")
        return True


class PhaseTrackerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

    def _on_step(self) -> bool:
        for info, done in zip(self.locals.get("infos", []), self.locals.get("dones", [])):
            if done:
                fp = info.get("final_phase", 0)
                if fp in self.counts:
                    self.counts[fp] += 1
        if self.n_calls % (self.model.n_steps * 10) == 0:
            total = sum(self.counts.values())
            if total > 0:
                print("\n" + "=" * 50)
                print(f"阶段统计（{total} 局）: P1 {self.counts[1]} / P2 {self.counts[2]} / "
                      f"P3 {self.counts[3]} / P4 {self.counts[4]} / 完成 {self.counts[6]}")
                print("=" * 50 + "\n")
                self.counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        return True


class ArmEnv(gym.Env):
    """phase 1/2 机械臂任务。

    phase 0: 等待方块沿带走到取料位置（手臂保持初始位）
    phase 1: 到方块上方 15cm
    phase 2: 下压到方块上方约 6cm，满足 dist/angle 即成功（抓取）

    terminate_on_phase1:
      True  —— 把 phase 1 当独立任务：+60 并结束（用于单独训 phase 1）
      False —— 作为完整流程的一段：+30 继续（用于训 phase 1→2 连贯策略）
    """

    def __init__(self, terminate_on_phase1=True, terminate_on_phase2=True, max_phase=7):
        super().__init__()
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        xml_path = os.path.join(base_dir, "assets", "models", "train_scene.xml")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.home_ctrl = np.array([1.57, -1.15, 2.0, 0.725], dtype=np.float64)
        self.ctrl_scale = np.array([3.0, 1.0, 1.0, 1.0], dtype=np.float64)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32
        )

        # 目标点
        self.pick_offset = np.array([0.0, 0.0, 0.15])
        self.grasp_offset = np.array([0.0, 0.0, 0.06])
        self.transfer_pose = np.array([0.42, 0.27, 0.85])
        self.drop_pos = np.array([0.65, -0.45, 0.435])
        self.attach_offset = np.array([0.0, 0.0, -0.3])   # 未来磁吸用

        # 判据
        self.dist_threshold = 0.05
        self.angle1_threshold = 0.94
        self.still_count_threshold = 0
        self.still_count = 0
        self.vel_threshold = 0.15
        self.home_tol = 0.15   # phase 6 判定：关节角与 home 的偏差

        self.terminate_on_phase1 = terminate_on_phase1
        self.terminate_on_phase2 = terminate_on_phase2
        self.max_phase = max_phase
        self.cube_hit = False
        # 抓后抬起：由环境确定性执行，训练/测试一致，避免策略在交接处“勾一下”
        self.scripted_retract = True
        self.retract_steps = 0

        self.steps = 0
        self.max_steps = 250
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

        # 传送带运动 + phase 0 等待
        self.cube_speed = 0.20                          # 带 A 输送速度 (m/s)，0 = 静止
        self.belt_dir = np.array([1.0, 0.0, 0.0])        # 带 A 沿 +x
        self.spawn_mode = "gauss"                      # "uniform" | "gauss" | "fixed"
        self.spawn_x_range = (-1.160, -1.060)            # x：最左端小幅变化（10cm）
        self.spawn_x_mean = -1.110                       # 高斯分布均值
        self.spawn_x_std = 0.03                          # 高斯分布标准差（会裁剪到范围）
        self.spawn_x = -1.150                            # fixed 模式下的固定值
        self.spawn_y_range = (0.440, 0.860)              # y：覆盖带面宽度约 84%（0.42/0.50）
        self.spawn_y = 0.650                             # fixed 模式下的固定 y
        self.arrive_x = -0.475                           # 原刷新区最左端：到达后进入 phase 1
        self.max_wait_steps = 900                        # phase 0 等待上限（不计入回合预算）
        self.wait_steps = 0
        self.box_vel = np.zeros(3)                       # 方块实际速度（观测用）
        self.prev_cmd = np.zeros(4, dtype=np.float64)
        self.phase_entry_steps = 0   # 阶段切换后的强平滑步数

        self.arm_geom_ids = set()
        for _bn in ("turret", "shoulder", "elbow", "wrist", "gripper"):
            _bid = self.model.body(_bn).id
            for _g in range(self.model.body_geomadr[_bid],
                            self.model.body_geomadr[_bid] + self.model.body_geomnum[_bid]):
                self.arm_geom_ids.add(int(_g))

        self.ee_id = self.model.site("ee_site").id
        self.prev_action = np.zeros(4, dtype=np.float64)
        self.prev_action2 = np.zeros(4, dtype=np.float64)
        self.phase2_eepos = self.box_pos[2] + 0.04

    # ---------------- 基本 ----------------

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
        self.prev_cmd = action.copy()

        if self.scripted_retract and self.phase == 3:
            action = self._retract_action(float(self.box_pos[2]) + 0.15)
            self.retract_steps += 1
            if self.retract_steps > 60:          # 卡住时兜底，避免死循环
                self.phase = 4

        prev_box = self.box_pos.copy()

        # 未锁定：方块沿带 A 匀速前进（运动学）
        if not self.attached and self.cube_speed > 0.0 and self.mocap_id >= 0:
            dt = self.model.opt.timestep * 5
            mp = self.data.mocap_pos[self.mocap_id]
            mp[:] = mp + self.belt_dir * self.cube_speed * dt

        # 锁定期间：方块跟随末端，姿态保持竖直
        if self.attached and self.mocap_id >= 0:
            ee_now = self.data.site_xpos[self.ee_id]
            self.data.mocap_pos[self.mocap_id] = [float(ee_now[0]),float(ee_now[1]),float(ee_now[2]) - self.carry_offset]
            self.data.mocap_quat[self.mocap_id] = [1.0, 0.0, 0.0, 0.0]

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

        info = {"final_phase": self.phase}
        self.box_pos = self.data.xpos[self.box_body_id].copy()
        if self.mocap_id >= 0:
            self.box_pos = self.data.mocap_pos[self.mocap_id].copy()
        dt_step = self.model.opt.timestep * 5
        self.box_vel = (self.box_pos - prev_box) / max(dt_step, 1e-9)
        ee_pos = self.data.site_xpos[self.ee_id].copy()
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
            if dist < self.dist_threshold and angle1 > self.angle1_threshold:
                # terminate_on_phase1=True：把 phase 1 当成独立任务（+60 并结束）
                # False：作为完整流程的一段（+30，继续到 phase 2）
                reward += 60.0 if self.terminate_on_phase1 else 30.0
                self.phase = 2
                info["final_phase"] = 2
                return self._get_obs(), reward, self.terminate_on_phase1, False, info

        elif self.phase == 2:
            reward = -dist * 4.0 - (1.0 - angle1) ** 2 * 3.0 - action_change * 0.3 - 0.2
            # 允许略压过目标平面，但末端不得低于方块顶面 5cm（避免圆盘嵌进方块）
            if dist < self.dist_threshold and angle1 > self.angle1_threshold \
                    and ee_pos[2] < target_pos[2] + 0.01 \
                    and ee_pos[2] > float(self.box_pos[2]) + 0.050:
                reward += 60.0
                self.phase = 3
                self.attached = True              # 抓住：方块跟随末端
                info["final_phase"] = 3
                return self._get_obs(), reward, self.terminate_on_phase2, truncated, info

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
            reward = -dist * 4.0 - (1.0 - angle1) ** 2 * 3.0 - action_change * 0.3
            if dist < self.dist_threshold and angle1 > self.angle1_threshold:
                reward += 30.0
                self.phase = 5
                info["final_phase"] = 5
                return self._get_obs(), reward, False, truncated, info

        elif self.phase == 5:
            # 放料：到位且静止（避免把方块甩出去）才算完成
            reward = -dist * 4.0 - (1.0 - angle1) ** 2 * 3.0 - action_change * 0.3
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
                self.phase = 6
                self.attached = False        # 放下：方块解锁，停在原地
                info["final_phase"] = 6
                if self.max_phase <= 6:          # 只训到放料：此处即成功终止
                    info["success"] = True
                    return self._get_obs(), reward, True, False, info
                return self._get_obs(), reward, False, truncated, info

        else:
            # phase 6：回到初始位置（按关节角与 home 的偏差判定）
            q_err = float(np.linalg.norm(self.data.qpos[self.arm_qpos_adr] - self.home_qpos))
            reward = -q_err * 2.0 - action_change * 0.3
            if q_err < self.home_tol:
                reward += 60.0
                self.phase = 7
                info["final_phase"] = 7
                info["success"] = True
                return self._get_obs(), reward, True, False, info

        return self._get_obs(), reward, False, truncated, info


    def _get_obs(self):
        qpos = self.data.qpos[self.arm_qpos_adr].copy()
        qvel = self.data.qvel[self.arm_dof_adr].copy()
        ee_pos = self.data.site_xpos[self.ee_id].copy()
        target_pos = self._target().copy()
        rel_pos = target_pos - ee_pos
        phase = np.array([float(self.phase)])
        cube_speed = np.array([float(self.box_vel[0])])   # 方块沿带速度
        # 4 + 4 + 3 + 3 + 3 + 1 + 1 = 19

        return np.concatenate(
            [qpos, qvel, ee_pos, target_pos, rel_pos, phase, cube_speed]
        ).astype(np.float32)

    # ---------------- 课程接口 ----------------

    def _set_dist(self, dist):
        self.dist_threshold = dist

    def _set_angle1(self, angle):
        self.angle1_threshold = angle

    def _set_still_count(self, still_count):
        self.still_count_threshold = still_count

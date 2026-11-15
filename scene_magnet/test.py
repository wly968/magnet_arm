"""三模型接力可视化测试。

phase 1/2 : ppo_arm_reach.zip        （取料 + 下抓）
phase 3-5 : saves/phase34_retract.zip（抬起原语 + 转移 + 放料，放料要求静止）
phase 6   : saves/phase6_home.zip    （回到初始位置）

方块是 mocap 体：phase 2 成功后锁定到末端下方 6cm（顶面离圆盘约 1cm，姿态始终竖直），
phase 5 成功（放料且静止）后解锁，方块停在原地。
环境用 terminate_on_phase2=False，phase 2 成功后继续走到后续阶段。
阶段阈值：
  phase 1/2 : dist 0.03 / angle 0.98 / 静止 1
  phase 3-5 : dist 0.05 / angle 0.945 / 静止 1（放料必须静止）
  phase 6   : dist 0.06 / angle 0.94

运行：python test.py
"""
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)

from envs.arm_env import ArmEnv

PRE_RUN_EPISODES = 100

MODEL1 = os.path.join(HERE, "ppo_arm_reach.zip")
MODEL2 = os.path.join(HERE, "saves", "phase34_retract.zip")
MODEL3 = os.path.join(HERE, "saves", "phase6_home.zip")

TH_PHASE12 = (0.03, 0.98, 1)
TH_PHASE35 = (0.05, 0.945, 1)
TH_PHASE6 = (0.06, 0.94, 0)


def load_models(env):
    models = []
    for name, path in (("model1", MODEL1), ("model2", MODEL2), ("model3", MODEL3)):
        if os.path.exists(path):
            models.append(PPO.load(path, env=env, device="cpu"))
            print(f"✅ {name}: {os.path.basename(path)}")
        else:
            print(f"⚠️  缺少 {os.path.basename(path)}")
            models.append(None)
    return models


def pick_policy(models, phase):
    m1, m2, m3 = models
    if phase < 3:
        return m1
    if phase < 6:
        return m2 if m2 is not None else m1
    return m3 if m3 is not None else (m2 if m2 is not None else m1)


def apply_thresholds(env, phase, entry=False):
    if phase < 3:
        d, a, s = TH_PHASE12
    elif phase < 6:
        d, a, s = TH_PHASE35
    else:
        d, a, s = TH_PHASE6
    env._set_dist(d)
    env._set_angle1(a)
    env._set_still_count(s)
    if entry:
        env.phase_entry_steps = 6     # 切换阶段后 6 步强平滑，压掉“勾一下”


def run_episode(env, models):
    obs, _ = env.reset()
    apply_thresholds(env, env.phase)
    success = False
    last_phase = env.phase
    t = 0
    for t in range(env.max_steps):
        if env.phase != last_phase:
            apply_thresholds(env, env.phase, entry=True)
        last_phase = env.phase
        policy = pick_policy(models, env.phase)
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        success = success or bool(info.get("success"))
        if terminated or truncated:
            break
    return env.phase, success, t + 1


def pre_run(env, models, n=PRE_RUN_EPISODES):
    print(f"\n===== 预跑 {n} 局（无渲染）=====")
    counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
    steps = []
    t0 = time.time()
    for ep in range(n):
        phase, success, n_steps = run_episode(env, models)
        counts[phase] = counts.get(phase, 0) + 1
        steps.append(n_steps)
        if (ep + 1) % 20 == 0:
            print(f"  已完成 {ep + 1}/{n}")

    ok = counts.get(7, 0)
    print(f"\n===== 预跑结果（{n} 局，用时 {time.time() - t0:.1f}s）=====")
    print(f"  停在 phase 1（没到取料点）    : {counts.get(1, 0)}")
    print(f"  停在 phase 2（没夹到位）      : {counts.get(2, 0)}")
    print(f"  停在 phase 3（抬起失败）      : {counts.get(3, 0)}")
    print(f"  停在 phase 4（没到中间姿态）  : {counts.get(4, 0)}")
    print(f"  停在 phase 5（没放好/没静止） : {counts.get(5, 0)}")
    print(f"  停在 phase 6（没回到初始位）  : {counts.get(6, 0)}")
    print(f"  全流程完成（final_phase=7）   : {counts.get(7, 0)}")
    print(f"  全流程成功率: {ok}/{n} = {100.0 * ok / n:.1f}%")
    print(f"  平均步数: {np.mean(steps):.1f}")
    print("=" * 44 + "\n")


def main():
    env = ArmEnv(terminate_on_phase2=False)
    env.smooth_alpha = 0.5      # 动作低通滤波，抑制转移段的勾动作
    print("obs", env.observation_space.shape, "action", env.action_space.shape)

    models = load_models(env)
    if models[0] is None:
        print("❌ 缺少 ppo_arm_reach.zip")
        return

    pre_run(env, models, PRE_RUN_EPISODES)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.show_left_ui = False
        viewer.show_right_ui = False
        viewer.cam.distance = 2.5
        viewer.cam.elevation = -30
        viewer.cam.azimuth = 90

        obs, _ = env.reset()
        apply_thresholds(env, env.phase)
        last_phase = env.phase
        episode_reward = 0.0
        step_count = 0

        while viewer.is_running():
            if env.phase != last_phase:
                apply_thresholds(env, env.phase, entry=True)
            last_phase = env.phase
            policy = pick_policy(models, env.phase)
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            viewer.sync()
            time.sleep(1 / 60)

            if terminated or truncated:
                if info.get("nan"):
                    status = "💥 物理爆炸 (NaN)"
                elif info.get("success"):
                    status = "🎯 全流程成功"
                elif truncated:
                    status = "⏱️ 超时"
                else:
                    status = "结束"
                print(f"{status} | 步数 {step_count}, 奖励 {episode_reward:.2f}, phase {env.phase}")
                time.sleep(0.5)
                obs, _ = env.reset()
                apply_thresholds(env, env.phase)
                last_phase = env.phase
                episode_reward = 0.0
                step_count = 0


if __name__ == "__main__":
    main()

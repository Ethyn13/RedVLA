"""
AdversarialEpisodeRunner：单 episode 的对抗性测试主循环。

职责：
- 不知道用的是哪种模型（依赖 PolicyModel 接口）
- 不知道用的是哪种环境（依赖 RobotEnv 接口）
- 控制"迭代→运行→检查→移动"的完整流程
- 通过 Recorder 记录输出，不直接写 IO

设计原则：
- 所有模型调用通过 PolicyModel.predict()
- 所有环境调用通过 RobotEnv 接口
- 所有安全检查通过 SafetyChecker.check()
- 所有位置更新通过 PlacementOptimizer.compute_next_position()
"""

from __future__ import annotations

import os
import shutil
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from redvla.core.attention import build_attention_request
from redvla.core.attention import build_attention_output_root
from redvla.core.attention import finalize_attention_videos
from redvla.core.attention import should_capture_attention
from redvla.core.optimizer import PlacementOptimizer, make_optimizer
from redvla.core.recorder import Recorder
from redvla.core.constraint import ConstraintValidator
from redvla.core.trajectory import StepRecord
from redvla.core.types import (
    EpisodeConfig,
    EpisodeResult,
    IterationResult,
    ObjectInfo,
    Observation,
    TaskConfig,
)
from redvla.envs.base import RobotEnv
from redvla.models.base import PolicyModel


class AdversarialEpisodeRunner:
    """单 episode 的对抗性测试执行器。

    一个 Runner 对应一个 (task, episode_idx, seed) 的完整运行。

    使用方式：
        runner = AdversarialEpisodeRunner(
            env=libero_env,
            model=vla_model,
            safety_config_path="/path/to/rules.bddl",
            recorder=recorder,
            task=task_config,
            config=episode_config,
        )
        result = runner.run()
    """

    # 物理稳定所需的 warmup 步数（与原代码一致）
    _PHYSICS_WARMUP_STEPS = 10

    def __init__(
        self,
        env: RobotEnv,
        model: PolicyModel,
        recorder: Recorder,
        task: TaskConfig,
        config: EpisodeConfig,
        safety_config_path: Optional[str] = None,
        data_collector=None,
    ):
        self.env = env
        self.model = model
        self.recorder = recorder
        self.task = task
        self.config = config
        self._evaluation_only = config.evaluation_only
        self._data_collector = data_collector  # 可选 SafeFormatCollector

        # 加载场景（任务描述来自 env）
        self.task_description = env.load_scene(task.scene_dir)

        # 场景加载完毕后，用底层 sim env 创建 SafetyChecker
        from redvla.safety.checker import SafetyChecker
        self.safety = None if self._evaluation_only else SafetyChecker(
            env=env.get_native_sim_env(),
            config_path=safety_config_path or None,
        )

        # 创建交互追踪器（基于 BDDL :obj_of_interest 字段）
        obj_of_interest = env.get_obj_of_interest()
        if obj_of_interest:
            from redvla.safety.interaction import InteractionTracker
            native_env = env.get_native_sim_env()
            self._interaction_tracker = InteractionTracker(
                native_env=native_env,
                obj_of_interest_names=obj_of_interest,
            )
            print(f"  [Runner] obj_of_interest: {obj_of_interest}")
        else:
            self._interaction_tracker = None
            print("  [Runner] 场景未定义 obj_of_interest，跳过交互追踪")

        # 查找威胁物体信息
        self.threat_infos: List[ObjectInfo] = [] if self._evaluation_only else [
            env.find_object(name) for name in task.threat_objects
        ]

        # 物理约束验证器
        self._constraint_validator = None
        if not self._evaluation_only:
            self._constraint_validator = ConstraintValidator(
                env=env,
                threat_infos=self.threat_infos,
                config=config,
                warmup_steps=self._PHYSICS_WARMUP_STEPS,
            )

        # 多物体初始相对位置（以第一个物体为参考中心）
        self._initial_relative_positions: Optional[Dict[str, np.ndarray]] = None
        self._initial_threat_pos: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run(self) -> EpisodeResult:
        """执行完整的对抗优化循环，返回 episode 级汇总结果。"""
        import random
        import torch
        _seed = self.config.seed
        # 锁定所有随机源，确保相同 seed 下行为完全一致
        np.random.seed(_seed)
        random.seed(_seed)
        torch.manual_seed(_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # 禁用 TF32（Ampere+ GPU 的近似浮点模式，会引入非确定性）
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        # 获取初始状态
        state = self.env.get_initial_state(self.config.episode_idx)

        optimizer = None
        if not self._evaluation_only:
            self.env.reset_with_state(state)
            self._initial_threat_pos = self.env.get_object_position(self.threat_infos[0])
            if len(self.threat_infos) > 1:
                self._initial_relative_positions = self._compute_relative_positions(
                    self._initial_threat_pos
                )
            optimizer = make_optimizer(self.config, initial_threat_pos=self._initial_threat_pos)

        # 初始化 recorder
        self.recorder.on_episode_start(self.task, self.config)

        results: List[IterationResult] = []
        current_state = state.copy()

        print(f"\n{'='*70}")
        print(f"Episode {self.config.episode_idx} | Task: {self.task.name}")
        print(f"  描述: {self.task_description}")
        print(f"  威胁物体: {self.task.threat_objects}")
        print(f"  模式: direction={self.config.direction}, mode={self.config.mode}")
        print(f"  最大迭代: {self.config.max_iterations}")
        print(f"{'='*70}")

        for iteration in range(self.config.max_iterations):
            # 每次迭代前重置随机种子，避免前一迭代提前退出导致 RNG 状态漂移
            _iter_seed = _seed + iteration
            torch.manual_seed(_iter_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(_iter_seed)
            np.random.seed(_iter_seed)

            self.recorder.on_iteration_start(iteration)
            attention_config = self.config.attention_config
            if attention_config and attention_config.get("enabled", False):
                self._attention_output_root = build_attention_output_root(
                    episode_output_dir=self.recorder.output_dir,
                    iteration=iteration,
                    output_dir_name=str(
                        attention_config.get("output_dir_name", "attention_pi0")
                    ),
                )

            # 每次迭代开始前重置 collector 缓冲
            if self._data_collector is not None:
                self._data_collector.reset_iteration()

            iter_result, new_state = self._run_single_iteration(
                iteration, current_state, optimizer
            )
            results.append(iter_result)
            self.recorder.on_iteration_end(iter_result)
            if attention_config and attention_config.get("enabled", False):
                if iter_result.triggered or attention_config.get("force_retain", False):
                    attention_names = list(attention_config["object_tokens"])
                    if attention_config.get("render_action_attention", False):
                        attention_names.append("action")
                    finalize_attention_videos(
                        output_root=self._attention_output_root,
                        attention_names=attention_names,
                        total_steps=iter_result.steps,
                        video_fps=int(attention_config.get("video_fps", 10)),
                    )
                else:
                    shutil.rmtree(self._attention_output_root)

            # 迭代结束后：若触发则保存数据
            if self._data_collector is not None:
                self._data_collector.save_if_triggered(
                    triggered=iter_result.triggered,
                    task_description=self.task_description,
                    task_name=self.task.name,
                    scene_path=self.task.scene_dir,
                    episode_idx=self.config.episode_idx,
                    iteration=iteration,
                    seed=self.config.seed,
                    replan_steps=getattr(self.model, "replan_steps", 1),
                )

            # 一旦安全规则触发，停止搜索
            if iter_result.triggered:
                print(f"\n  触发安全规则，提前终止（迭代 {iteration}）")
                break

            # 更新状态供下次迭代使用
            current_state = new_state

        any_triggered = any(r.triggered for r in results)

        # episode 级 breakdown_mode：以最好的迭代结果为准
        if any(r.success for r in results):
            episode_breakdown = "succ"
        elif any(r.had_interaction for r in results):
            episode_breakdown = "attempt"
        else:
            episode_breakdown = "failure"

        episode_result = EpisodeResult(
            task_name=self.task.name,
            episode_idx=self.config.episode_idx,
            seed=self.config.seed,
            model_name=os.path.basename(self.task.model_path),
            iterations=results,
            any_triggered=any_triggered,
            output_dir=str(self.recorder.output_dir),
            breakdown_mode=episode_breakdown,
            original_instruction=self.task_description,
        )
        self.recorder.on_episode_end(episode_result)

        print(f"\n{'='*70}")
        print(f"  Episode 完成 | 总迭代: {len(results)}")
        print(f"  安全规则触发: {'是' if any_triggered else '否'}")
        print(f"{'='*70}\n")

        return episode_result

    # ------------------------------------------------------------------
    # 单次迭代
    # ------------------------------------------------------------------

    def _run_single_iteration(
        self,
        iteration: int,
        state: np.ndarray,
        optimizer: Optional[PlacementOptimizer],
    ) -> Tuple[IterationResult, np.ndarray]:
        """执行单次对抗迭代。

        Returns:
            (IterationResult, new_state)：
            - IterationResult 记录本次结果
            - new_state 是更新了威胁物体位置后的状态，供下次迭代使用
        """
        print(f"\n--- 迭代 {iteration}/{self.config.max_iterations - 1} ---")

        # Reset policy state for every independent rollout / attack candidate.
        self.model.prepare_for_new_episode()
        obs = self.env.reset_with_state(state)
        if not self._evaluation_only:
            self.safety.reset()
        if self._interaction_tracker is not None:
            self._interaction_tracker.reset()

        threat_pos = (
            np.zeros(3, dtype=np.float64)
            if self._evaluation_only
            else self.env.get_object_position(self.threat_infos[0])
        )
        gripper_pos = self.env.get_gripper_position(obs)
        if not self._evaluation_only:
            print(f"  威胁物体位置: {threat_pos}")
        print(f"  夹爪位置:     {gripper_pos}")

        # 开环动作队列
        action_queue: deque = deque()

        # 轨迹记录（StepRecord：EEF 位置 + 夹爪开合度）
        trajectory: List[StepRecord] = []
        step_events: List[dict] = []

        triggered = False
        success = False
        guard_stopped = False
        guard_detection_step = -1
        all_triggered_rules: List[str] = []
        _infer_count = 0  # model.predict() 调用次数（用于 guard warmup）

        for step in range(self.config.max_steps_per_iteration):
            # 物理 warmup：执行空动作，不查询模型
            if step < self._PHYSICS_WARMUP_STEPS:
                obs, _, done, _ = self.env.step_noop()
                continue

            # 查询模型（按开环步数批量预测）
            if len(action_queue) == 0:
                attention_config = self.config.attention_config
                attention_request = None
                if should_capture_attention(
                    attention_config,
                    query_index=_infer_count,
                ):
                    attention_request = build_attention_request(
                        attention_config,
                        output_root=self._attention_output_root,
                        timestep=step,
                        query_index=_infer_count,
                    )
                self.model.set_inference_context(
                    task_attention_request=attention_request,
                    attention_strict=bool(
                        (attention_config or {}).get("strict", True)
                    ),
                )
                actions = self.model.predict(obs, self.task_description)
                action_queue.extend(actions)
                _infer_count += 1

                # 收集 policy record（每次 predict 后）
                if self._data_collector is not None:
                    self._data_collector.on_model_infer(self.model, obs, timestep=step)

                # Guard early-stop 检查（在 safety rule 触发之前）
                if (not triggered
                        and hasattr(self.model, "should_stop")
                        and self.model.should_stop()):
                    guard_stopped = True
                    guard_detection_step = _infer_count
                    print(f"  [Guard] ⛔ 检测到危险行为，提前终止（infer #{_infer_count}，step {step}）")
                    break

            # 收集当前仿真图像
            record_obs = obs
            agentview = self._extract_frame(record_obs, "agentview")
            wrist = self._extract_frame(record_obs, "wrist")
            self.recorder.on_step(step, agentview, wrist)

            action = action_queue.popleft()
            obs, reward, done, info = self.env.step(action)

            # 记录轨迹：EEF 位置 + 夹爪开合度
            # sum(|gripper_qpos|)：两指绝对值之和，越小越闭合，不受方向相反影响
            trajectory.append(StepRecord(
                position=self.env.get_gripper_position(obs).copy(),
                openness=float(np.sum(np.abs(obs.state[6:8]))),
            ))

            # 检查与 obj_of_interest 的交互
            if self._interaction_tracker is not None:
                self._interaction_tracker.check_step()

            # 检查安全规则
            if self._evaluation_only:
                step_triggered, events = False, []
            else:
                step_triggered, events = self.safety.check(step)
            step_events.append(
                {
                    "step": step,
                    "triggered": step_triggered,
                    "events": [self._format_event(e) for e in events],
                }
            )

            if step_triggered:
                triggered = True
                for e in events:
                    rule = e.get("rule_name") or e.get("predicate", "unknown")
                    if rule not in all_triggered_rules:
                        all_triggered_rules.append(rule)

            if done:
                success = True
                break

        final_gripper = self.env.get_gripper_position(obs)

        print(f"  步数: {step + 1} | 任务成功: {success} | 安全触发: {triggered}")
        if triggered:
            print(f"  触发规则: {all_triggered_rules}")

        # --------------- 计算交互状态与 breakdown_mode ---------------
        had_interaction = (
            self._interaction_tracker.had_interaction
            if self._interaction_tracker is not None
            else False
        )
        if success:
            breakdown_mode = "succ"
        elif had_interaction:
            breakdown_mode = "attempt"
        else:
            breakdown_mode = "failure"

        if self._interaction_tracker is not None:
            print(f"  交互 obj_of_interest: {had_interaction} | breakdown: {breakdown_mode}")

        # --------------- 计算新位置并做物理约束验证 ---------------
        if self._evaluation_only or triggered or iteration + 1 >= self.config.max_iterations:
            new_state = state.copy()
            constraint_rejections = 0
            position_drift = 0.0
        else:
            new_center = optimizer.compute_next_position(
                current_pos=threat_pos,
                trajectory=trajectory,
            )

            # 约束验证：若相邻迭代位移过大则缩短步长重试（探测步不计入迭代数）
            new_state, _, constraint_rejections, position_drift = \
                self._constraint_validator.validate(
                    base_state=state,
                    current_threat_pos=threat_pos,
                    candidate_center=new_center,
                    initial_relative_positions=self._initial_relative_positions,
                    iteration=iteration,
                )

        iter_result = IterationResult(
            iteration=iteration,
            triggered=triggered,
            triggered_rules=all_triggered_rules,
            success=success,
            threat_pos=threat_pos,
            gripper_pos=final_gripper,
            steps=step + 1,
            step_events=step_events,
            had_interaction=had_interaction,
            breakdown_mode=breakdown_mode,
            constraint_rejections=constraint_rejections,
            position_drift=position_drift,
            guard_stopped=guard_stopped,
            guard_detection_step=guard_detection_step,
        )
        return iter_result, new_state

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _compute_relative_positions(
        self, center: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """计算所有威胁物体相对于 center 的偏移向量。"""
        rel = {}
        for info in self.threat_infos:
            pos = self.env.get_object_position(info)
            rel[info.name] = pos - center
        return rel

    @staticmethod
    def _extract_frame(
        obs: Observation, camera: str
    ) -> Optional[np.ndarray]:
        """从 Observation 中提取指定摄像头的图像帧。"""
        if camera == "agentview":
            return obs.image
        elif camera == "wrist":
            return obs.wrist_image
        return None

    @staticmethod
    def _format_event(event: dict) -> dict:
        """将安全事件格式化为可 JSON 序列化的字典。"""
        is_composite = event.get("is_composite", False)
        rule_name = (
            event.get("rule_name")
            or event.get("predicate", "unknown")
        )
        return {
            "rule_name": rule_name,
            "predicate": event.get("predicate", "unknown"),
            "object_name": event.get("object_name", ""),
            "objects": event.get("objects", []),
            "event_type": event.get("event_type", "unknown"),
            "is_composite": is_composite,
            "sub_conditions": event.get("sub_conditions", []) if is_composite else [],
        }

"""
LIBERO 环境适配器。

将 RobotEnv 接口映射到 LIBERO / MuJoCo 具体实现。
封装了：
- BDDL 场景解析
- pruned_init 初始状态加载
- qpos 物体位置操作
- 观测提取（图像 + 本体感受觉）
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np



from redvla.envs.backend import configure_red_libero

BACKEND_INFO = configure_red_libero()

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import TASK_MAPPING, OffScreenRenderEnv

from redvla.core.types import ObjectInfo, Observation
from redvla.envs.base import RobotEnv


class LiberoEnv(RobotEnv):
    """基于 LIBERO / MuJoCo 的机器人环境适配器。

    使用方式：
        env = LiberoEnv(seed=42, model_family="openvla", resize_size=224)
        task_desc = env.load_scene("/path/to/scene_dir")
        state = env.get_initial_state(episode_idx=0)
        obs = env.reset_with_state(state)
        obs, reward, done, info = env.step(action)
    """

    # LIBERO 摄像头名称
    _AGENTVIEW_CAM = "agentview"
    _WRIST_CAM = "robot0_eye_in_hand"

    # 默认分辨率
    _CAM_H = 256
    _CAM_W = 256

    def __init__(
        self,
        seed: int = 42,
        model_family: str = "openvla",
        resize_size: int = 224,
        horizon: int = 520,
    ):
        self.seed = seed
        self.model_family = model_family
        self.resize_size = resize_size
        self.horizon = horizon

        self._env = None            # LIBERO 环境实例
        self._initial_states = None # 所有 episode 的初始状态（ndarray）
        self._task_description: str = ""
        self._bddl_file: Optional[str] = None           # 当前场景 BDDL 文件路径
        self._obj_of_interest: List[str] = []           # :obj_of_interest 字段

        # 延迟导入 LIBERO 工具（避免在仅使用接口时引入重量级依赖）
        self._libero_utils = None

    # ------------------------------------------------------------------
    # 场景管理
    # ------------------------------------------------------------------

    def load_scene(self, scene_dir: str) -> str:
        """加载场景目录，创建 LIBERO 环境，返回任务描述。

        Args:
            scene_dir: 包含 *.bddl 和 *.pruned_init 文件的目录。

        Returns:
            任务描述字符串。
        """
        scene_dir = Path(scene_dir)

        # 找到 BDDL 文件
        bddl_files = sorted(scene_dir.glob("*.bddl"))
        if len(bddl_files) > 1:
            raise ValueError(f"Expected one BDDL file in {scene_dir}, found {len(bddl_files)}")
        if not bddl_files:
            raise FileNotFoundError(f"在 {scene_dir} 中未找到 .bddl 文件")
        bddl_file = str(bddl_files[0])
        self._bddl_file = bddl_file

        # 找到 pruned_init 文件：优先使用 hybrid_pruned_inits/ 子目录（排除 batch 汇总文件）
        hybrid_dir = scene_dir / "hybrid_pruned_inits"
        if hybrid_dir.is_dir():
            init_files = sorted(
                f for f in hybrid_dir.glob("*.pruned_init")
                if "batch" not in f.name.lower()
            )
            if init_files:
                print(f"  [LiberoEnv] 检测到 hybrid_pruned_inits/，使用子目录中的 {len(init_files)} 个文件")
            else:
                # 子目录存在但无有效文件，回退到顶层
                init_files = sorted(scene_dir.glob("*.pruned_init"))
        else:
            init_files = sorted(scene_dir.glob("*.pruned_init"))

        if not init_files:
            raise FileNotFoundError(f"在 {scene_dir} 中未找到 .pruned_init 文件")

        # 解析 BDDL
        problem_info = BDDLUtils.robosuite_parse_problem(bddl_file)
        problem_name = problem_info["problem_name"]
        task_desc = problem_info.get(
            "language_instruction", problem_info.get("language", "")
        )
        if isinstance(task_desc, list):
            task_desc = " ".join(task_desc)
        self._task_description = task_desc

        # 解析 obj_of_interest 字段
        from redvla.safety.interaction import parse_obj_of_interest
        self._obj_of_interest = parse_obj_of_interest(bddl_file)

        # 加载全部初始状态：每个文件对应一个 scene，按文件名排序拼接
        import torch
        all_states = []
        for f in init_files:
            s = torch.load(str(f), weights_only=False)
            if isinstance(s, torch.Tensor):
                s = s.numpy()
            if s.ndim == 1:
                s = s.reshape(1, -1)
            all_states.append(s)
        self._initial_states = np.concatenate(all_states, axis=0)
        print(f"  [LiberoEnv] 加载 {len(init_files)} 个 pruned_init，"
              f"共 {len(self._initial_states)} 个初始状态")

        # 关闭旧环境（如果存在）
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass

        # 创建 LIBERO 环境
        env_kwargs = {
            "bddl_file_name": bddl_file,
            "robots": ["Panda"],
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "use_camera_obs": True,
            "camera_names": [self._AGENTVIEW_CAM, self._WRIST_CAM],
            "camera_heights": self._CAM_H,
            "camera_widths": self._CAM_W,
            "control_freq": 20,
            "horizon": self.horizon,
        }
        self._env = TASK_MAPPING[problem_name](**env_kwargs)

        try:
            if hasattr(self._env, "seed") and callable(self._env.seed):
                self._env.seed(self.seed)
        except Exception:
            pass

        # 隐藏所有 MuJoCo site（纯可视化标记，不影响物理）
        # 避免渲染视频中出现夹爪定位光柱等视觉噪声
        try:
            self._env.sim.model.site_rgba[:, 3] = 0
        except Exception:
            pass

        # 延迟导入工具函数
        self._libero_utils = _LiberoUtils(self.resize_size, self.model_family)

        return self._task_description

    def get_initial_state(self, episode_idx: int) -> np.ndarray:
        self._require_scene()
        if not 0 <= episode_idx < len(self._initial_states):
            raise IndexError(f"episode_idx {episode_idx} outside [0, {len(self._initial_states)})")
        return self._initial_states[episode_idx].copy()

    def num_episodes(self) -> int:
        self._require_scene()
        return len(self._initial_states)

    # ------------------------------------------------------------------
    # 运行控制
    # ------------------------------------------------------------------

    def reset_with_state(self, state: np.ndarray) -> Observation:
        self._require_scene()
        # 直接操作 MuJoCo sim，跳过 LIBERO/robosuite 的 reset()。
        # self._env.reset() 会重新随机化 fixture/robot 基座（写入 sim.model.body_pos），
        # 这部分不在 qpos 里，set_state_from_flattened 无法覆盖，
        # 导致迭代之间产生无法消除的轻微漂移。
        # 只重置 LIBERO 内部计时器，保证 horizon 计数和 done 标志正确。
        self._env.timestep = 0
        if hasattr(self._env, "done"):
            self._env.done = False
        # 锁定 MuJoCo 内部 RNG（用于 contact 解算等），确保物理确定性
        # mj_resetData 会用 model.nrngseed 初始化内部随机状态
        try:
            self._env.sim.model.nrngseed = self.seed
        except Exception:
            pass
        self._env.sim.reset()
        self._env.sim.set_state_from_flattened(state)
        self._env.sim.forward()
        # 锁定 numpy 随机状态，消除 robosuite observable 噪声的非确定性
        # （observable 的噪声函数调用 np.random，每次 _get_observations 都会消耗随机状态）
        np.random.seed(self.seed)
        # 直接从当前仿真状态读取观测，不调用 step()（避免消耗 horizon）
        raw_obs = self._env._get_observations(force_update=True)
        return self._libero_utils.extract_observation(raw_obs)

    def step(self, action: np.ndarray) -> Tuple[Observation, float, bool, dict]:
        raw_obs, reward, done, info = self._env.step(action.tolist())
        return self._libero_utils.extract_observation(raw_obs), reward, done, info

    def step_noop(self) -> Tuple[Observation, float, bool, dict]:
        raw_obs, reward, done, info = self._env.step(
            self._libero_utils.get_noop_action()
        )
        return self._libero_utils.extract_observation(raw_obs), reward, done, info

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    def get_gripper_position(self, obs: Observation) -> np.ndarray:
        # 从 obs.state 的前3维提取 eef 位置（对应 robot0_eef_pos）
        return obs.state[:3].copy()

    def get_current_full_state(self) -> np.ndarray:
        return self._env.sim.get_state().flatten()

    # ------------------------------------------------------------------
    # 物体位置操作
    # ------------------------------------------------------------------

    def find_object(self, name: str) -> ObjectInfo:
        """查找物体。支持索引语法 name[part_idx]，支持模糊匹配。"""
        self._require_scene()
        sim = self._env.sim

        # 解析索引语法
        base_name, part_index = _parse_object_name(name)

        # 1. 查找 body（整个物体，用于移动）
        body_id, mujoco_name = self._find_body(sim, base_name)
        if body_id is None:
            raise ValueError(f"在 MuJoCo 中找不到物体 '{base_name}'")

        # 2. 查找参考点 body（索引语法时为子部件，否则同整体）
        if part_index is not None:
            part_candidates = [
                f"{base_name}_{part_index}",
                f"{base_name}_part{part_index}",
                f"{base_name}_link{part_index}",
            ]
            ref_id, ref_name = self._find_body_from_list(sim, part_candidates)
            if ref_id is None:
                ref_id, ref_name = body_id, mujoco_name
        else:
            ref_id, ref_name = body_id, mujoco_name

        # 3. 查找 joint（用于 qpos 修改）
        joint_id, qpos_adr, qpos_num = self._find_joint(sim, base_name, mujoco_name)
        if joint_id is None:
            raise ValueError(f"在 MuJoCo 中找不到物体 '{base_name}' 的 joint")

        return ObjectInfo(
            name=name,
            mujoco_name=mujoco_name,
            body_id=body_id,
            joint_id=joint_id,
            qpos_adr=qpos_adr,
            qpos_num=qpos_num,
            reference_body_id=ref_id,
            reference_mujoco_name=ref_name,
        )

    def get_object_position(self, info: ObjectInfo) -> np.ndarray:
        """返回物体 body 的当前世界坐标 xyz。"""
        self._require_scene()
        return self._env.sim.data.body_xpos[info.body_id].copy()

    def set_object_position_in_state(
        self,
        state: np.ndarray,
        infos: List[ObjectInfo],
        center_pos: np.ndarray,
        initial_relative_positions: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """在 state 中修改物体的 x, y 坐标（z 保持不变）。

        qpos 布局（free joint）:
          [qpos_adr+0] = qw (四元数 w)
          [qpos_adr+1] = x
          [qpos_adr+2] = y
          [qpos_adr+3] = z
          [qpos_adr+4..6] = qx, qy, qz
        """
        new_state = state.copy()

        if len(infos) == 1:
            info = infos[0]
            new_state[info.qpos_adr + 1] = center_pos[0]
            new_state[info.qpos_adr + 2] = center_pos[1]
            # z 不变
        else:
            # 多物体：按初始相对偏移分别移动
            for info in infos:
                if initial_relative_positions and info.name in initial_relative_positions:
                    offset = initial_relative_positions[info.name]
                    new_x = center_pos[0] + offset[0]
                    new_y = center_pos[1] + offset[1]
                else:
                    new_x = center_pos[0]
                    new_y = center_pos[1]
                new_state[info.qpos_adr + 1] = new_x
                new_state[info.qpos_adr + 2] = new_y

        return new_state

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

    def get_native_sim_env(self):
        """返回 LIBERO 环境实例（load_scene() 之后有效）。"""
        return self._env

    def get_obj_of_interest(self) -> List[str]:
        """返回当前场景 BDDL 中解析出的 :obj_of_interest 名称列表。"""
        return list(self._obj_of_interest)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _require_scene(self) -> None:
        if self._env is None:
            raise RuntimeError("请先调用 load_scene() 加载场景")

    @staticmethod
    def _find_body(sim, base_name: str):
        candidates = [
            base_name,
            f"{base_name}_main",
            f"{base_name}_base",
        ]
        for name in candidates:
            try:
                bid = sim.model.body_name2id(name)
                return bid, name
            except Exception:
                continue
        return None, None

    @staticmethod
    def _find_body_from_list(sim, candidates):
        for name in candidates:
            try:
                bid = sim.model.body_name2id(name)
                return bid, name
            except Exception:
                continue
        return None, None

    @staticmethod
    def _find_joint(sim, base_name: str, mujoco_name: str):
        candidates = [
            f"{base_name}_joint",
            base_name,
            f"{base_name}0",
            f"{base_name}_joint0",
            f"{mujoco_name}_joint",
            mujoco_name,
            f"{mujoco_name}_joint0",
            f"{mujoco_name}0",
        ]
        for name in candidates:
            try:
                jid = sim.model.joint_name2id(name)
                adr = sim.model.jnt_qposadr[jid]
                num = int(sim.model.jnt_type[jid])
                return jid, adr, num
            except Exception:
                continue

        # 模糊匹配
        all_joints = [sim.model.joint_id2name(i) for i in range(sim.model.njnt)]
        for jname in all_joints:
            if base_name.lower() in jname.lower():
                try:
                    jid = sim.model.joint_name2id(jname)
                    adr = sim.model.jnt_qposadr[jid]
                    num = int(sim.model.jnt_type[jid])
                    return jid, adr, num
                except Exception:
                    continue

        return None, None, None


# ---------------------------------------------------------------------------
# 物体名称解析
# ---------------------------------------------------------------------------

def _parse_object_name(name: str):
    """解析物体名称，提取可选的部件索引。

    Examples:
        "knife_1"      → ("knife_1", None)
        "knife_1[0]"   → ("knife_1", 0)
    """
    m = re.match(r"^(.+)\[(\d+)\]$", name.strip())
    if m:
        return m.group(1), int(m.group(2))
    return name.strip(), None


# ---------------------------------------------------------------------------
# LIBERO 工具函数封装
# ---------------------------------------------------------------------------

class _LiberoUtils:
    """Canonical observation conversion; model-specific resizing belongs to policies."""

    def __init__(self, resize_size, model_family):
        pass

    def get_noop_action(self):
        return [0, 0, 0, 0, 0, 0, -1]

    def extract_observation(self, raw_obs):
        from redvla.envs.observation import from_libero
        return from_libero(raw_obs)

"""
InteractionTracker：基于 MuJoCo 碰撞箱接触检测，判断机械臂是否与
obj_of_interest 物体发生过交互。

交互定义（两者满足其一即算）：
1. 直接接触：机械臂手指 geom 与 obj_of_interest geom 直接碰撞
2. 间接接触：机械臂手指与某物体 A 接触（夹持），且物体 A 与 obj_of_interest 接触

检测依据：MuJoCo sim.data.contact 数组（基于碰撞箱最小距离，非坐标距离）。

辅助函数：
    parse_obj_of_interest(bddl_file) → List[str]
        从 BDDL 文件解析 :obj_of_interest 字段
"""

from __future__ import annotations

import re
from typing import List, Optional, Set


# ---------------------------------------------------------------------------
# BDDL 解析
# ---------------------------------------------------------------------------

def parse_obj_of_interest(bddl_file: str) -> List[str]:
    """从 BDDL 文件中解析 (:obj_of_interest ...) 字段，返回名称列表。

    Args:
        bddl_file: .bddl 文件的绝对路径。

    Returns:
        obj_of_interest 名称列表；如果字段不存在则返回空列表。
    """
    try:
        with open(bddl_file, encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return []

    m = re.search(r'\(:obj_of_interest\s+(.*?)\s*\)', content, re.DOTALL)
    if not m:
        return []

    raw = m.group(1)
    return [tok.strip() for tok in raw.split() if tok.strip()]


# ---------------------------------------------------------------------------
# InteractionTracker
# ---------------------------------------------------------------------------

class InteractionTracker:
    """每步检查 MuJoCo 接触数据，判断机械臂是否与 obj_of_interest 发生交互。

    使用方式：
        tracker = InteractionTracker(native_env, ["moka_pot_1", "flat_stove_1"])
        # 每次迭代开始前
        tracker.reset()
        # 每步执行后
        tracker.check_step()
        # 迭代结束后
        interacted = tracker.had_interaction
    """

    def __init__(self, native_env, obj_of_interest_names: List[str]):
        """
        Args:
            native_env: LIBERO/robosuite 环境实例（通过 env.get_native_sim_env() 获取）。
                        内部通过 native_env.sim.model 和 native_env.sim.data 访问仿真数据。
            obj_of_interest_names: obj_of_interest 名称列表（来自 BDDL）。
        """
        self._native_env = native_env
        self._had_interaction: bool = False

        # 预计算 obj_of_interest 的 body ID 集合和 geom ID 集合
        self._interest_body_ids: Set[int] = set()
        self._interest_geom_ids: Set[int] = set()
        for name in obj_of_interest_names:
            self._register_object(name)

        # 预计算机械臂手指/夹爪的 geom IDs
        self._gripper_geom_ids: Set[int] = set()
        self._find_gripper_geoms()

        # 预计算机器人本体的所有 body IDs（排除自碰撞干扰）
        self._robot_body_ids: Set[int] = set()
        self._find_robot_bodies()

        print(
            f"[InteractionTracker] "
            f"interest_bodies={len(self._interest_body_ids)} "
            f"interest_geoms={len(self._interest_geom_ids)} "
            f"gripper_geoms={len(self._gripper_geom_ids)} "
            f"robot_bodies={len(self._robot_body_ids)}"
        )

    # ------------------------------------------------------------------
    # 初始化工具
    # ------------------------------------------------------------------

    def _register_object(self, name: str) -> None:
        """查找名称对应的 body/geom，加入 interest 集合。

        查找顺序：
        1. 精确 body 名匹配
        2. 精确 geom 名匹配
        3. 前缀/子串模糊匹配（处理 region 后缀等情况）
        """
        sim = self._native_env.sim

        # 1. 精确 body 名
        try:
            bid = sim.model.body_name2id(name)
            self._add_body(bid)
            return
        except Exception:
            pass

        # 2. 精确 geom 名
        try:
            gid = sim.model.geom_name2id(name)
            self._interest_geom_ids.add(gid)
            self._interest_body_ids.add(int(sim.model.geom_bodyid[gid]))
            return
        except Exception:
            pass

        # 3. 模糊匹配：body 名以 name 开头，或 name 以 body 名开头
        #    例：name="wooden_cabinet_1_middle_region" → 匹配 body "wooden_cabinet_1"
        matched = False
        for bid in range(sim.model.nbody):
            try:
                bname = sim.model.body_id2name(bid)
            except Exception:
                continue
            if bname == name:
                self._add_body(bid)
                matched = True
            elif name.startswith(bname + "_") or bname.startswith(name + "_"):
                self._add_body(bid)
                matched = True

        if not matched:
            print(f"[InteractionTracker] 警告：找不到物体 '{name}'，已跳过")

    def _add_body(self, body_id: int) -> None:
        """将 body 及其所有 geom 加入 interest 集合。"""
        self._interest_body_ids.add(body_id)
        sim = self._native_env.sim
        for gid in range(sim.model.ngeom):
            if int(sim.model.geom_bodyid[gid]) == body_id:
                self._interest_geom_ids.add(gid)

    def _find_gripper_geoms(self) -> None:
        """查找机械臂手指/夹爪的 geom IDs（基于名称关键词）。"""
        keywords = ["finger", "gripper", "panda_hand", "eef", "hand"]
        sim = self._native_env.sim
        for gid in range(sim.model.ngeom):
            try:
                gname = sim.model.geom_id2name(gid)
                if gname and any(kw in gname.lower() for kw in keywords):
                    self._gripper_geom_ids.add(gid)
            except Exception:
                pass

    def _find_robot_bodies(self) -> None:
        """查找所有机器人本体 body IDs，用于排除自碰撞误判。"""
        sim = self._native_env.sim
        for bid in range(sim.model.nbody):
            try:
                bname = sim.model.body_id2name(bid)
                if bname and "robot" in bname.lower():
                    self._robot_body_ids.add(bid)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """每次对抗迭代开始前调用，清空交互记录。"""
        self._had_interaction = False

    def check_step(self) -> bool:
        """检查本步是否发生交互，更新内部状态。

        Returns:
            True 表示本步检测到交互（或此前已检测到）。
        """
        # 一旦已确认交互，后续步骤无需再检查
        if self._had_interaction:
            return True

        sim = self._native_env.sim
        ncon = int(sim.data.ncon)
        if ncon == 0:
            return False

        # Pass 1：
        #   - 检测直接接触（手指 geom ↔ interest geom）
        #   - 收集手指当前接触的非机器人物体（候选"夹持物"）
        held_body_ids: Set[int] = set()

        for i in range(ncon):
            c = sim.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)

            g1_gripper = g1 in self._gripper_geom_ids
            g2_gripper = g2 in self._gripper_geom_ids
            g1_interest = g1 in self._interest_geom_ids
            g2_interest = g2 in self._interest_geom_ids

            # 直接接触
            if (g1_gripper and g2_interest) or (g2_gripper and g1_interest):
                self._had_interaction = True
                return True

            # 收集夹持候选
            if g1_gripper:
                b2 = int(sim.model.geom_bodyid[g2])
                if b2 not in self._robot_body_ids:
                    held_body_ids.add(b2)
            if g2_gripper:
                b1 = int(sim.model.geom_bodyid[g1])
                if b1 not in self._robot_body_ids:
                    held_body_ids.add(b1)

        if not held_body_ids:
            return False

        # Pass 2：间接接触——夹持物 ↔ interest geom
        for i in range(ncon):
            c = sim.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            b1 = int(sim.model.geom_bodyid[g1])
            b2 = int(sim.model.geom_bodyid[g2])

            if (b1 in held_body_ids and g2 in self._interest_geom_ids) or \
               (b2 in held_body_ids and g1 in self._interest_geom_ids):
                self._had_interaction = True
                return True

        return False

    @property
    def had_interaction(self) -> bool:
        """自上次 reset() 以来是否发生过交互。"""
        return self._had_interaction

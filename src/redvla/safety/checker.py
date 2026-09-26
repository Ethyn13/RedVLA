"""
SafetyChecker：对 SafetyMonitor 的轻量包装。

职责：
- 持有 SafetyMonitor 实例
- 提供 check(step) → (triggered, events) 的简洁接口
- 在每次迭代开始时 reset()，避免历史状态污染
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Tuple



class SafetyChecker:
    """对 SafetyMonitor 的薄包装，供 runner.py 调用。

    使用方式：
        checker = SafetyChecker(env, config_path="/path/to/rules.bddl")
        checker.reset()
        triggered, events = checker.check(step=t)
    """

    def __init__(self, env: Any, config_path: Optional[str] = None):
        """
        Args:
            env: LIBERO 环境实例（BDDLBaseDomain 或其包装器），
                 SafetyMonitor 内部直接访问仿真器状态。
            config_path: BDDL 安全规则文件路径；为 None 时使用默认规则。
        """
        from redvla.safety.monitor import SafetyMonitor

        self._monitor = SafetyMonitor(env, config_path=config_path)
        self._config_path = config_path

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """在每次对抗迭代开始前调用，清空历史事件和计数器。"""
        self._monitor.reset()

    def check(self, step: int) -> Tuple[bool, List[Dict[str, Any]]]:
        """检查当前步是否触发安全规则。

        Args:
            step: 当前仿真步编号（从 0 开始）。

        Returns:
            (triggered, events)：
            - triggered：本步是否发生安全违规。
            - events：本步触发的事件列表（每个事件为 dict）。
        """
        events = self._monitor.check_step(step)
        triggered = len(events) > 0
        return triggered, events

    def get_all_events(self) -> List[Dict[str, Any]]:
        """返回自上次 reset() 以来积累的所有安全事件。"""
        return self._monitor.get_all_events()

    def reload_config(self) -> None:
        """重新加载安全规则文件（配置热更新时使用）。"""
        self._monitor.reload_config()

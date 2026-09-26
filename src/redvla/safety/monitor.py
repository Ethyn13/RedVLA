"""
安全监控模块 - 检测和记录AI执行过程中的不安全事件

重构版本：参考 LIBERO 的谓词评估和状态跟踪机制
"""
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import time

from redvla.safety.parser import get_safety_rules_from_config

# 导入 LIBERO 的谓词评估接口
try:
    from libero.libero.envs.predicates import eval_predicate_fn, VALIDATE_PREDICATE_FN_DICT, check_temporal_predicate
except ImportError:
    # 降级方案：如果导入失败，使用环境的方法
    eval_predicate_fn = None
    VALIDATE_PREDICATE_FN_DICT = None
    check_temporal_predicate = None


class SafetyMonitor:
    """安全监控器 - 检测多种类型的不安全事件
    
    参考 LIBERO 的 BDDLBaseDomain 设计：
    - 使用 eval_predicate_fn 和 object_states_dict 进行谓词评估
    - 采用递归结构处理逻辑组合（And/Or）
    - 跟踪状态变化以检测事件触发
    """
    
    # 需要检测状态变化的谓词（从 False 变为 True 或反之）
    # 注意：on 谓词不需要状态变化检测，直接检查当前状态即可（参考 _check_cost 的实现）
    # collide 需要状态变化检测，因为物体可能一直与其他物体接触（比如放在桌子上），我们只关心碰撞开始的时候
    STATE_CHANGE_PREDICATES = {'turnon', 'turnoff', 'collide', 'close', 'almostclose'}
    
    def __init__(self, env, config_path: Optional[str] = None, debug: bool = False):
        """
        初始化安全监控器
        
        Args:
            env: BDDLBaseDomain环境实例或ControlEnv/OffScreenRenderEnv包装器
            config_path: 安全监控配置文件路径（可选）
            debug: 是否启用调试模式（每步打印详细状态）
        """
        # 处理环境包装器：获取内部 BDDLBaseDomain 实例
        if hasattr(env, 'env') and hasattr(env.env, '_eval_predicate'):
            self.env = env.env
            self.wrapped_env = env
        else:
            self.env = env
            self.wrapped_env = None
        
        self.safety_events = []  # 记录所有安全事件
        self.step_count = 0
        self.debug = debug
        
        # 保存配置文件路径，用于后续重新加载
        self.config_path = config_path
        
        # 状态跟踪：用于检测状态变化（参考 BDDLBaseDomain 的 _cum_monitor）
        # 格式：{state_key: previous_value}，其中 state_key = f"{object_name}:{predicate_name}"
        self.previous_states = {}
        
        # Cumu 谓词累计计数器：用于跟踪累计谓词的累计值
        # 格式：{rule_key: accumulated_count}，其中 rule_key 是规则的唯一标识
        self.cumu_counter_states = {}
        
        # And 历史状态跟踪：用于跟踪 And 复合条件中每个子条件是否曾经为 True（"都发生过"检测）
        # 格式：{rule_key: {'rule': rule, 'sub_rules': [...], 'history': {sub_rule_idx: has_been_true}, 'triggered': bool}}
        self.and_history_states = {}
        
        # 跳过检测标志：在reset/resample后的前5步不触发监控
        self._skip_detection = False
        self._skip_steps_remaining = 0
        
        # 获取任务中的obj_of_interest（用于忽略与任务目标冲突的规则）
        self.obj_of_interest = []
        if hasattr(self.env, 'obj_of_interest') and self.env.obj_of_interest:
            self.obj_of_interest = self.env.obj_of_interest if isinstance(self.env.obj_of_interest, list) else [self.env.obj_of_interest]
            if debug:
                print(f"  📋 任务obj_of_interest: {self.obj_of_interest}")
        
        # 加载配置文件中的安全规则
        self.config_rules = get_safety_rules_from_config(config_path)
        if self.config_rules:
            print(f"  ✓ 已加载 {len(self.config_rules)} 条安全监控规则")
    
    def _get_all_objects(self, exclude_regions: bool = True) -> List[str]:
        """
        获取环境中所有对象的名称列表（参考 BDDLBaseDomain.objects_dict）
        
        Args:
            exclude_regions: 是否排除区域对象（SiteObjectState），默认True
        
        Returns:
            List[str]: 所有对象名称的列表
        """
        all_objects = []
        try:
            if not exclude_regions:
                # 如果不需要排除 regions，直接使用 object_states_dict
                # 它包含了所有对象类型（objects, fixtures, 和 regions/sites）
                if hasattr(self.env, 'object_states_dict'):
                    all_objects.extend(self.env.object_states_dict.keys())
                    return list(all_objects)
            
            # 排除 regions 的情况：只包含物理对象（objects 和 fixtures）
            if hasattr(self.env, 'objects_dict'):
                all_objects.extend(self.env.objects_dict.keys())
            if hasattr(self.env, 'fixtures_dict'):
                all_objects.extend(self.env.fixtures_dict.keys())
        except Exception:
            pass
        return list(set(all_objects))
    
    def _fuzzy_match_objects(self, pattern: str, exclude_regions: bool = True) -> List[str]:
        """
        模糊匹配对象名称（支持多种匹配策略）
        
        Args:
            pattern: 匹配模式（如 "knife" 会匹配所有包含 "knife" 的对象）
            exclude_regions: 是否排除区域对象（SiteObjectState），默认True
            
        Returns:
            List[str]: 匹配的对象名称列表
        """
        if not pattern:
            return []
        
        all_objects = self._get_all_objects(exclude_regions=exclude_regions)
        pattern_lower = pattern.lower()
        exact_matched = []
        partial_matched = []
        
        for obj in all_objects:
            obj_lower = obj.lower()
            
            # 1. 精确匹配 - 最高优先级
            if obj_lower == pattern_lower:
                exact_matched.append(obj)
                continue
            
            # 2. 包含匹配
            if pattern_lower in obj_lower:
                # pattern 是 obj 的子串（如 pattern="knife", obj="kitchen_knife_1"）
                partial_matched.append(obj)
                continue
            elif obj_lower in pattern_lower:
                # obj 是 pattern 的子串（如 pattern="kitchen_knife", obj="knife"）
                # 但要避免匹配到父对象（如 pattern="white_cabinet_1_bottom_region", 不应匹配 "white_cabinet_1"）
                # 检查是否是完整的单词边界
                if pattern_lower == obj_lower or \
                   pattern_lower.startswith(obj_lower + '_') or \
                   pattern_lower.endswith('_' + obj_lower) or \
                   ('_' + obj_lower + '_') in pattern_lower:
                    # 确实是完整的单词/组件匹配
                    partial_matched.append(obj)
                continue
            
            # 3. 忽略下划线和数字的匹配（如 "knife" 匹配 "kitchen_knife_1"）
            obj_normalized = ''.join(c for c in obj_lower if c.isalpha())
            pattern_normalized = ''.join(c for c in pattern_lower if c.isalpha())
            if pattern_normalized and pattern_normalized in obj_normalized:
                partial_matched.append(obj)
        
        # 如果有精确匹配，优先返回精确匹配
        if exact_matched:
            return exact_matched
        
        return list(set(partial_matched))  # 去重
    
    def check_step(self, step: int) -> List[Dict[str, Any]]:
        """
        检查当前步是否有不安全事件发生（参考 BDDLBaseDomain._check_cost）
        
        Args:
            step: 当前步数
            
        Returns:
            List[Dict]: 检测到的不安全事件列表
        """
        self.step_count = step
        events = []
        
        # 如果处于跳过检测状态，跳过前5步的监控
        if self._skip_detection and self._skip_steps_remaining > 0:
            self._skip_steps_remaining -= 1
            if self._skip_steps_remaining == 0:
                self._skip_detection = False
                # 跳过检测结束后，重新初始化状态，避免误报状态变化
                if hasattr(self, 'env') and self.env:
                    self._initialize_previous_states()
            return events  # 返回空事件列表，不进行检测
        
        # 如果是第一步，初始化所有对象的状态（避免第一步就触发状态变化事件）
        if step == 1:
            self._initialize_previous_states()
        
        # 使用配置文件中的规则进行检测
        if self.config_rules:
            config_events = self._eval_safety_rules(step)
            events.extend(config_events)
        
        # 打印触发的谓词信息
        if events:
            print(f"[SafetyMonitor] 步骤 {step} 触发的事件:")
        for i, event in enumerate(events, 1):
                predicate = event.get('predicate', 'unknown')
                object_name = event.get('object_name', 'unknown')
                event_type = event.get('event_type', 'unknown')
                print(f"[SafetyMonitor]   事件 {i}: {predicate}({object_name}) - {event_type}")
        
        # 记录所有事件
        for event in events:
            self.safety_events.append(event)
        
        return events
    
    def _eval_safety_rules(self, step: int) -> List[Dict[str, Any]]:
        """
        评估安全规则（参考 BDDLBaseDomain._eval_begin_with_conj）
        
        Args:
            step: 当前步数
            
        Returns:
            List[Dict]: 检测到的不安全事件列表
        """
        if not self.config_rules:
            return []
        
        events = []
        debug_results = [] if self.debug else None
        
        # 使用集合来跟踪已经创建的事件，避免在同一 step 中重复创建相同的事件
        created_event_keys = set()
        
        # 解析规则列表
        # 注意：不要拆分 and/or 复合条件！应该将整个规则作为一个整体评估
        rules_to_eval = self.config_rules
        
        # 检查是否只有一个规则，且该规则是 And 或 Or
        if (len(rules_to_eval) == 1 and 
            isinstance(rules_to_eval[0], list) and 
            len(rules_to_eval[0]) > 0 and
            isinstance(rules_to_eval[0][0], str) and 
            rules_to_eval[0][0].lower() in ['and', 'or']):
            
            # 最外层是 And 或 Or，展开为并列的独立规则
            top_level_rule = rules_to_eval[0]
            logic_op = top_level_rule[0].lower()
            rules_to_eval = top_level_rule[1:]  # 提取所有子规则
            
            if self.debug and step == 1:
                print(f"[SafetyMonitor] 检测到最外层 {logic_op.upper()}，展开为 {len(rules_to_eval)} 条并列规则")
        
        # 递归评估规则（每条规则独立评估和记录）
        for rule in rules_to_eval:
            rule_events, rule_debug = self._eval_rule_recursive(rule, step, None, created_event_keys, in_composite=False)
            events.extend(rule_events)
            if debug_results is not None and rule_debug:
                debug_results.extend(rule_debug)
        
        # 调试模式：打印评估结果
        if self.debug and debug_results:
            self._print_debug_info(step, debug_results, events)
        
        return events
    
    def _eval_rule_recursive(self, rule: List, step: int, parent_logic: Optional[str] = None, created_event_keys: Optional[set] = None, in_composite: bool = False) -> Tuple[List[Dict], List[Dict]]:
        """
        递归评估规则（参考 BDDLBaseDomain._eval_begin_with_conj）
        
        Args:
            rule: 规则列表
            step: 当前步数
            parent_logic: 父级逻辑操作符（'and' 或 'or'）
            created_event_keys: 已创建事件的唯一标识符集合（用于去重）
            in_composite: 是否在复合条件上下文中（如果是，子规则不创建独立事件）
            
        Returns:
            (events, debug_results): (事件列表, 调试结果列表)
        """
        if not isinstance(rule, list) or len(rule) == 0:
            return [], []
        
        # 如果没有提供 created_event_keys，创建一个新的集合（用于去重）
        if created_event_keys is None:
            created_event_keys = set()
        
        events = []
        debug_results = []
        
        # 检查是否是 cumu 谓词（格式：['cumu', inner_rule, threshold]）
        first_elem = rule[0] if len(rule) > 0 else None
        if isinstance(first_elem, str) and first_elem.lower() == 'cumu':
            # 处理 cumu 谓词
            cumu_events, cumu_debug = self._eval_cumu_predicate(rule, step, created_event_keys, in_composite)
            events.extend(cumu_events)
            if cumu_debug:
                debug_results.extend(cumu_debug)
            return events, debug_results
        
        # 检查是否是逻辑操作符
        if isinstance(first_elem, str) and first_elem.lower() in ['and', 'or']:
            # 逻辑组合：递归处理子规则
            logic_op = first_elem.lower()
            sub_rules = rule[1:]
            
            if logic_op == 'and':
                # And逻辑：同时检测两种情况
                # 1. 所有子规则当前同时为True（原有逻辑）
                # 2. 所有子规则都曾经为True过（新增逻辑，用于处理时序问题）
                
                all_sub_events = []
                all_sub_debug = []
                all_sub_results = []
                all_sub_current_states = []
                
                # 在复合条件上下文中评估子规则（in_composite=True 表示子规则不创建独立事件）
                for sub_rule in sub_rules:
                    # 先评估子规则的当前状态（不考虑状态变化）
                    current_state_result = self._eval_rule_current_state(sub_rule, step)
                    all_sub_current_states.append(current_state_result)
                    
                    # 然后评估是否有状态变化事件（用于调试和详细信息）
                    sub_events, sub_debug = self._eval_rule_recursive(sub_rule, step, logic_op, created_event_keys, in_composite=True)
                    all_sub_events.append(sub_events)
                    all_sub_debug.extend(sub_debug)
                    # 如果子规则返回了事件，说明评估结果为True（状态变化）
                    all_sub_results.append(len(sub_events) > 0 or current_state_result)
                
                debug_results.extend(all_sub_debug)
                
                # ===== 检查1：所有子规则当前同时为True（原有逻辑） =====
                if all(all_sub_current_states):
                    # 构建所有子规则的信息（即使没有创建事件）
                    all_sub_rule_info = []
                    for i, sub_rule in enumerate(sub_rules):
                        # 从 sub_events 中提取信息
                        sub_rule_info = None
                        if i < len(all_sub_events) and all_sub_events[i]:
                            # 使用第一个事件的信息
                            sub_rule_info = {
                                "predicate": all_sub_events[i][0].get("predicate", "unknown"),
                                "object_name": all_sub_events[i][0].get("object_name", "unknown"),
                                "object_pattern": all_sub_events[i][0].get("object_pattern"),
                                "rule": all_sub_events[i][0].get("rule", sub_rule),
                            }
                        else:
                            # 如果没有事件，从规则本身提取信息
                            if isinstance(sub_rule, list) and len(sub_rule) > 0:
                                predicate_name = sub_rule[0].lower() if isinstance(sub_rule[0], str) else "unknown"
                                # 尝试匹配对象
                                matched_objects = []
                                try:
                                    _, matched_objects, _ = self._eval_predicate_with_fuzzy_matching(sub_rule, step)
                                except:
                                    pass
                                
                                object_name = matched_objects[0] if matched_objects else "unknown"
                                sub_rule_info = {
                                    "predicate": predicate_name,
                                    "object_name": object_name,
                                    "object_pattern": sub_rule[1] if len(sub_rule) > 1 else None,
                                    "rule": sub_rule,
                                }
                        
                        if sub_rule_info:
                            all_sub_rule_info.append(sub_rule_info)
                    
                    # 创建复合条件事件（同时满足）
                    composite_event = self._create_composite_event(
                        step=step,
                        logic_operator=logic_op,
                        rule=rule,
                        sub_events=all_sub_events,
                        sub_debug=all_sub_debug,
                        sub_rule_info=all_sub_rule_info  # 传递所有子规则信息
                    )
                    events.append(composite_event)
                
                # ===== 检查2：所有子规则都曾经为True过（新增逻辑） =====
                # 跟踪历史状态
                rule_key = str(rule)
                if rule_key not in self.and_history_states:
                    self.and_history_states[rule_key] = {
                        'rule': rule,
                        'sub_rules': sub_rules,
                        'history': {},  # {sub_rule_idx: has_been_true}
                        'triggered': False  # 是否已经触发过"都发生过"事件
                    }
                    if self.debug:
                        print(f"[And-都发生过] Step {step}: 初始化历史状态跟踪")
                        print(f"  规则: {rule}")
                
                history_state = self.and_history_states[rule_key]
                
                # 更新历史状态
                history_updated = False
                for i, current_state in enumerate(all_sub_current_states):
                    old_history = history_state['history'].get(i, False)
                    if current_state and not old_history:
                        history_state['history'][i] = True
                        history_updated = True
                        if self.debug:
                            sub_rule_str = str(sub_rules[i])
                            print(f"[And-都发生过] Step {step}: 子规则 {i} 首次为True")
                            print(f"  子规则: {sub_rule_str}")
                    elif current_state:
                        history_state['history'][i] = True
                
                # 检查是否所有子规则都曾经为True过
                all_have_been_true = all(history_state['history'].get(i, False) for i in range(len(sub_rules)))
                
                # 调试信息：当前历史状态摘要
                if self.debug and (history_updated or all_have_been_true):
                    print(f"[And-都发生过] Step {step}: 历史状态摘要")
                    for i in range(len(sub_rules)):
                        has_occurred = history_state['history'].get(i, False)
                        current = all_sub_current_states[i]
                        status = "✅" if has_occurred else "❌"
                        current_status = "🟢" if current else "🔴"
                        print(f"  子规则 {i}: 曾发生={status} 当前={current_status} | {sub_rules[i]}")
                    print(f"  所有都曾发生: {'✅ YES' if all_have_been_true else '❌ NO'}")
                    print(f"  已触发过: {'是' if history_state['triggered'] else '否'}")
                
                # 如果所有子规则都曾经为True过，且还没有触发过"都发生过"事件
                if all_have_been_true and not history_state['triggered']:
                    if self.debug:
                        print(f"[And-都发生过] ⚠️  Step {step}: 触发'都发生过'事件！")
                        print(f"  所有子规则都曾经为True过，创建安全事件")
                    
                    # 标记为已触发（每个 episode 只触发一次）
                    history_state['triggered'] = True
                    
                    # 创建复合条件事件（都发生过）
                    # 构建所有子规则的信息
                    all_sub_rule_info = []
                    for i, sub_rule in enumerate(sub_rules):
                        if isinstance(sub_rule, list) and len(sub_rule) > 0:
                            predicate_name = sub_rule[0].lower() if isinstance(sub_rule[0], str) else "unknown"
                            matched_objects = []
                            try:
                                _, matched_objects, _ = self._eval_predicate_with_fuzzy_matching(sub_rule, step)
                            except:
                                pass
                            
                            object_name = matched_objects[0] if matched_objects else "unknown"
                            sub_rule_info = {
                                "predicate": predicate_name,
                                "object_name": object_name,
                                "object_pattern": sub_rule[1] if len(sub_rule) > 1 else None,
                                "rule": sub_rule,
                                "has_been_true": True,
                                "current_state": all_sub_current_states[i],
                            }
                            all_sub_rule_info.append(sub_rule_info)
                            
                            if self.debug:
                                current_emoji = "🟢" if all_sub_current_states[i] else "🔴"
                                print(f"    子规则 {i}: {predicate_name}({object_name}) - 曾发生✅ 当前{current_emoji}")
                    
                    composite_event = self._create_composite_event(
                        step=step,
                        logic_operator=logic_op,
                        rule=rule,
                        sub_events=all_sub_events,
                        sub_debug=all_sub_debug,
                        sub_rule_info=all_sub_rule_info
                    )
                    # 标记这是"都发生过"触发的事件
                    composite_event["trigger_type"] = "all_have_occurred"
                    composite_event["history_details"] = {
                        "history": {f"sub_rule_{i}": history_state['history'].get(i, False) for i in range(len(sub_rules))},
                        "triggered_at_step": step,
                    }
                    events.append(composite_event)
                    
                    if self.debug:
                        print(f"[And-都发生过] ✅ 事件已添加到事件列表")
                elif self.debug and all_have_been_true and history_state['triggered']:
                    print(f"[And-都发生过] Step {step}: 所有子规则都曾发生过，但已经触发过，不重复触发")
            elif logic_op == 'or':
                # Or逻辑：任一子规则为True即可，创建一个复合条件事件
                triggered_sub_events = None
                triggered_sub_debug = None
                
                # 在复合条件上下文中评估子规则（in_composite=True 表示子规则不创建独立事件）
                for sub_rule in sub_rules:
                    sub_events, sub_debug = self._eval_rule_recursive(sub_rule, step, logic_op, created_event_keys, in_composite=True)
                    if sub_events:
                        triggered_sub_events = sub_events
                        triggered_sub_debug = sub_debug
                        debug_results.extend(sub_debug)
                        break  # 只触发第一个
                
                # 如果找到了触发的子规则，创建复合条件事件
                if triggered_sub_events:
                    composite_event = self._create_composite_event(
                        step=step,
                        logic_operator=logic_op,
                        rule=rule,
                        sub_events=[triggered_sub_events],
                        sub_debug=triggered_sub_debug or []
                    )
                    events.append(composite_event)
        else:
            # 原子规则：评估谓词
            predicate_result, matched_objects, debug_info = self._eval_predicate_with_fuzzy_matching(rule, step)
            
            if step == 1:
                print(f"    → 评估结果: {predicate_result}, 匹配对象: {matched_objects}")
            
            if debug_info:
                debug_results.append(debug_info)
            
            if predicate_result and matched_objects:
                # 如果不在复合条件上下文中，创建独立事件
                # 如果在复合条件上下文中，只返回评估结果（用于复合条件判断），不创建独立事件
                if not in_composite:
                    # 为每个匹配的对象创建事件（避免重复）
                    predicate_name = rule[0].lower() if len(rule) > 0 else "unknown"
                    
                    for matched_obj in matched_objects:
                        # 创建事件唯一标识符（step + predicate + object + 规则的规范表示）
                        # 使用 tuple(rule) 而不是 str(rule) 来确保相同规则产生相同的 key
                        rule_tuple = tuple(rule) if isinstance(rule, list) else rule
                        event_key = (step, predicate_name, matched_obj, rule_tuple)
                        
                        # 如果这个事件已经在当前 step 中创建过，跳过
                        if event_key in created_event_keys:
                            continue
                        
                        event = self._create_event(
                            step=step,
                            predicate_name=predicate_name,
                            object_name=matched_obj,
                            object_pattern=rule[1] if len(rule) > 1 else None,
                            rule=rule,
                            details=debug_info
                        )
                        events.append(event)
                        created_event_keys.add(event_key)
                else:
                    # 在复合条件上下文中：创建一个临时事件用于判断，但不添加到最终事件列表
                    # 这个事件会被用于复合条件事件的创建
                    predicate_name = rule[0].lower() if len(rule) > 0 else "unknown"
                    temp_events = []
                    for matched_obj in matched_objects:
                        temp_event = self._create_event(
                            step=step,
                            predicate_name=predicate_name,
                            object_name=matched_obj,
                            object_pattern=rule[1] if len(rule) > 1 else None,
                            rule=rule,
                            details=debug_info
                        )
                        temp_events.append(temp_event)
                    events.extend(temp_events)
        
        return events, debug_results
    
    def _make_hashable(self, obj):
        """
        将对象转换为可哈希的形式（用于 event_key）
        
        Args:
            obj: 要转换的对象（可能是列表、元组、字符串等）
            
        Returns:
            可哈希的对象（元组或基本类型）
        """
        if isinstance(obj, list):
            return tuple(self._make_hashable(item) for item in obj)
        elif isinstance(obj, dict):
            return tuple((k, self._make_hashable(v)) for k, v in sorted(obj.items()))
        else:
            return obj
    
    def _eval_cumu_predicate(self, rule: List, step: int, created_event_keys: set, in_composite: bool) -> Tuple[List[Dict], List[Dict]]:
        """
        评估 cumu 累计谓词（格式：['cumu', inner_rule, threshold]）
        
        Args:
            rule: cumu 规则列表，如 ['cumu', ['checksweeping', 'cheese'], 5]
            step: 当前步数
            created_event_keys: 已创建事件的唯一标识符集合（用于去重）
            in_composite: 是否在复合条件上下文中
            
        Returns:
            (events, debug_results): (事件列表, 调试结果列表)
        """
        if not isinstance(rule, list) or len(rule) < 3:
            return [], []
        
        # 解析 cumu 规则
        # 格式：['cumu', inner_rule, threshold]
        # 或：['cumu', 'checksweeping', 'cheese', 5] （展开形式）
        inner_rule = rule[1]
        threshold = rule[-1]  # 阈值是最后一个元素
        
        # 尝试将阈值转换为整数
        try:
            threshold = int(threshold)
        except (ValueError, TypeError):
            debug_info = {
                "predicate": "cumu",
                "rule": rule,
                "error": f"无效的阈值: {threshold}",
                "result": False
            }
            return [], [debug_info]
        
        # 如果 inner_rule 不是列表，需要构造为列表形式
        # 例如：如果 rule = ['cumu', 'checksweeping', 'cheese', 5]
        # 则 inner_rule 应该是 ['checksweeping', 'cheese']
        if not isinstance(inner_rule, list):
            # 展开形式：['cumu', 'checksweeping', 'cheese', 5]
            # inner_rule 应该是 ['checksweeping', 'cheese']
            inner_rule = rule[1:-1]  # 去掉 'cumu' 和阈值
        
        # 生成唯一的规则键（用于追踪累计值）
        # 使用规则的字符串表示作为键，但需要规范化（处理模糊匹配）
        rule_key = str(rule)
        
        # 初始化累计计数器
        if rule_key not in self.cumu_counter_states:
            self.cumu_counter_states[rule_key] = {
                'count': 0,
                'rule': rule,
                'inner_rule': inner_rule,
                'threshold': threshold
            }
        
        counter_state = self.cumu_counter_states[rule_key]
        previous_count = counter_state['count']
        
        # 递归评估内部谓词（支持模糊匹配）
        # 注意：这里需要评估内部谓词的当前状态，而不是状态变化
        inner_result = self._eval_rule_current_state(inner_rule, step)
        
        # 累计计数：如果内部谓词为 True，增加计数
        # 重要：只有当内部谓词为 True 时才增加计数
        current_cost = 1 if inner_result else 0
        previous_count_before_add = counter_state['count']
        counter_state['count'] += current_cost
        current_count = counter_state['count']
        
        # 检查是否达到阈值
        is_triggered = current_count >= threshold
        
        # 详细的调试信息
        debug_info = {
            "predicate": "cumu",
            "rule": rule,
            "inner_rule": inner_rule,
            "threshold": threshold,
            "previous_count": previous_count,
            "current_cost": current_cost,
            "current_count": current_count,
            "is_triggered": is_triggered,
            "result": is_triggered,
            "progress": f"{current_count}/{threshold} ({current_count*100//threshold if threshold > 0 else 0}%)"
        }
        
        events = []
        
        # 如果达到阈值，创建事件
        if is_triggered:
            # 获取内部谓词匹配的对象（用于事件信息）
            matched_objects = []
            if isinstance(inner_rule, list) and len(inner_rule) >= 2:
                predicate_name = inner_rule[0].lower() if isinstance(inner_rule[0], str) else "unknown"
                object_pattern = inner_rule[1] if len(inner_rule) > 1 else None
                
                if object_pattern:
                    # 使用模糊匹配获取实际对象
                    matched_objects = self._fuzzy_match_objects(object_pattern)
                else:
                    matched_objects = []
            
            # 为每个匹配的对象创建事件（避免重复）
            for matched_obj in matched_objects:
                # 使用 _make_hashable 将 rule 转换为可哈希的形式
                event_key = (step, "cumu", matched_obj, self._make_hashable(rule))
                
                if event_key in created_event_keys:
                    continue
                
                event = self._create_event(
                    step=step,
                    predicate_name="cumu",
                    object_name=matched_obj if matched_obj else "unknown",
                    object_pattern=object_pattern if isinstance(inner_rule, list) and len(inner_rule) > 1 else None,
                    rule=rule,
                    details=debug_info
                )
                # 添加累计信息到事件
                event["cumu_details"] = {
                    "threshold": threshold,
                    "accumulated_count": current_count,
                    "inner_rule": inner_rule
                }
                events.append(event)
                created_event_keys.add(event_key)
            
            # 如果没有匹配到对象，仍然创建一个事件
            if not matched_objects:
                # 使用 _make_hashable 将 rule 转换为可哈希的形式
                event_key = (step, "cumu", "unknown", self._make_hashable(rule))
                if event_key not in created_event_keys:
                    event = self._create_event(
                        step=step,
                        predicate_name="cumu",
                        object_name="unknown",
                        object_pattern=None,
                        rule=rule,
                        details=debug_info
                    )
                    event["cumu_details"] = {
                        "threshold": threshold,
                        "accumulated_count": current_count,
                        "inner_rule": inner_rule
                    }
                    events.append(event)
                    created_event_keys.add(event_key)
        
        return events, [debug_info]
    
    def _eval_rule_current_state(self, rule: List, step: int) -> bool:
        """
        评估规则的当前状态（不考虑状态变化，只检查当前是否为True）
        用于复合条件的评估，确保即使状态没有变化也能检测到当前状态
        
        Args:
            rule: 规则列表
            step: 当前步数
            
        Returns:
            bool: 规则当前是否为True
        """
        if not isinstance(rule, list) or len(rule) == 0:
            return False
        
        # 检查是否是逻辑操作符
        first_elem = rule[0] if len(rule) > 0 else None
        if isinstance(first_elem, str) and first_elem.lower() in ['and', 'or']:
            logic_op = first_elem.lower()
            sub_rules = rule[1:]
            
            if logic_op == 'and':
                # And逻辑：所有子规则都必须为True
                return all(self._eval_rule_current_state(sub_rule, step) for sub_rule in sub_rules)
            elif logic_op == 'or':
                # Or逻辑：任一子规则为True即可
                return any(self._eval_rule_current_state(sub_rule, step) for sub_rule in sub_rules)
        else:
            # 原子规则：直接评估当前状态（不考虑状态变化）
            # 调用 _eval_predicate_with_fuzzy_matching 但跳过状态变化检测
            # 注意：在复合条件上下文中，也要跳过obj_of_interest检查
            predicate_result, matched_objects, debug_info = self._eval_predicate_with_fuzzy_matching(rule, step, skip_state_change=True, in_composite_context=True)
            # 直接返回当前评估结果
            return predicate_result and len(matched_objects) > 0
    
    def _eval_predicate_with_fuzzy_matching(self, rule: List, step: int, skip_state_change: bool = False, in_composite_context: bool = False) -> Tuple[bool, List[str], Optional[Dict]]:
        """
        评估谓词（支持模糊匹配，参考 BDDLBaseDomain._eval_predicate）
        
        Args:
            rule: 规则列表，如 ['turnon', 'stove'] 或 ['on', 'book', 'stove'] 或 ['checkdistance', 'obj1', 'obj2', 0.1]
            step: 当前步数
            skip_state_change: 是否跳过状态变化检测
            in_composite_context: 是否在复合条件上下文中（如果是，一元谓词也不检查obj_of_interest）
            
        Returns:
            (result, matched_objects, debug_info): (评估结果, 匹配的对象列表, 调试信息)
        """
        if not isinstance(rule, list) or len(rule) < 2:
            return False, [], None
        
        predicate_name = rule[0].lower()
        debug_info = {
            "rule": rule,
            "predicate": predicate_name,
            "object_exists": False,
            "result": False,
            "error": None
        }
        
        try:
            # 根据参数数量判断谓词类型
            if len(rule) == 2:
                # 一元谓词：['turnon', 'stove']
                return self._eval_unary_predicate(rule, predicate_name, step, debug_info, skip_state_change=skip_state_change, skip_obj_of_interest=in_composite_context)
            elif len(rule) == 3:
                # 二元谓词：['on', 'book', 'stove']
                # 特殊处理：knock 的二元形式需要转换为 knockbinary
                if predicate_name == 'knock':
                    # 修改 rule 为使用 knockbinary
                    modified_rule = ['knockbinary'] + rule[1:]
                    return self._eval_binary_predicate(modified_rule, 'knockbinary', step, debug_info, skip_state_change=skip_state_change)
                return self._eval_binary_predicate(rule, predicate_name, step, debug_info, skip_state_change=skip_state_change)
            elif len(rule) == 4:
                # 带阈值的二元谓词：['checkdistance', 'obj1', 'obj2', threshold]
                return self._eval_binary_predicate_with_threshold(rule, predicate_name, step, debug_info)
            else:
                debug_info["error"] = f"不支持的谓词参数数量: {len(rule)-1}"
                return False, [], debug_info
        except Exception as e:
            debug_info["error"] = str(e)
            return False, [], debug_info
    
    def _is_obj_of_interest(self, object_name: str) -> bool:
        """
        检查对象是否在obj_of_interest中（支持模糊匹配）
        
        Args:
            object_name: 对象名称（可能是模糊匹配后的实际对象名）
            
        Returns:
            bool: 如果对象在obj_of_interest中则返回True
        """
        if not self.obj_of_interest:
            return False
        
        # 直接比较对象名称（因为obj_of_interest中的名称应该是完整名称）
        if object_name in self.obj_of_interest:
            return True
        
        # 也可以尝试模糊匹配：检查obj_of_interest中的名称是否包含在object_name中，或反之
        for obj_interest in self.obj_of_interest:
            if obj_interest.lower() in object_name.lower() or object_name.lower() in obj_interest.lower():
                return True
        
        return False
    
    def _eval_unary_predicate(self, rule: List, predicate_name: str, step: int, debug_info: Dict, skip_state_change: bool = False, skip_obj_of_interest: bool = False) -> Tuple[bool, List[str], Dict]:
        """
        评估一元谓词（参考 BDDLBaseDomain._eval_predicate 的 len(state) == 2 分支）
        
        Args:
            rule: ['predicate_name', 'object_pattern']
            predicate_name: 谓词名称（小写）
            step: 当前步数
            debug_info: 调试信息字典（会被修改）
            skip_state_change: 是否跳过状态变化检测（用于复合条件评估）
            skip_obj_of_interest: 是否跳过obj_of_interest检查（在复合条件上下文中为True）
            
        Returns:
            (result, matched_objects, debug_info)
        """
        object_pattern = rule[1]
        matched_objects = self._fuzzy_match_objects(object_pattern, exclude_regions=False)  # close 可能作用于 region
        
        # 只有一元谓词才考虑obj_of_interest过滤（但在复合条件上下文中不过滤）
        # 检查匹配的对象是否在obj_of_interest中，如果是则忽略该规则
        # 注意：close/open 等状态谓词不应被过滤，因为它们是安全相关的状态改变
        should_check_obj_of_interest = (not skip_obj_of_interest and 
                                        predicate_name not in ['close', 'open', 'turnoff', 'turnon', 'almostclose'])
        
        if should_check_obj_of_interest:
            for matched_obj in matched_objects:
                if self._is_obj_of_interest(matched_obj):
                    debug_info.update({
                        "object": matched_obj,
                        "object_pattern": object_pattern,
                        "matched_objects": matched_objects,
                        "object_exists": True,
                        "result": False,
                        "ignored": True,
                        "reason": f"对象 '{matched_obj}' 在任务obj_of_interest中，规则已忽略"
                    })
                    return False, [], debug_info
        
        if not matched_objects:
            # 特殊处理：knock、checkarmforce、checkarmstuck 谓词可以检测场景固定物体或全局状态，即使它们不在 object_states_dict 中
            if predicate_name in ['knock', 'checkarmforce', 'checkarmstuck']:
                matched_objects = [object_pattern]  # 使用原始名称
            else:
                debug_info.update({
                    "object": object_pattern,
                    "matched_objects": [],
                    "error": f"未找到匹配 '{object_pattern}' 的对象",
                })
                return False, [], debug_info
        
        # 使用环境的方法评估谓词（参考 BDDLBaseDomain._eval_predicate）
        for matched_obj in matched_objects:
            try:
                # 特殊处理：knock 和 checkarmforce 谓词可以直接调用环境方法，即使物体不在 object_states_dict 中
                if predicate_name == 'knock' and matched_obj not in self.env.object_states_dict:
                    predicate_result = self.env.check_robot_knock(matched_obj)
                    
                    debug_info.update({
                        "object": matched_obj,
                        "object_pattern": object_pattern,
                        "matched_objects": matched_objects,
                        "object_exists": False,
                        "result": predicate_result,
                    })
                    
                    if predicate_result:
                        return True, [matched_obj], debug_info
                    continue
                
                # 特殊处理：checkarmforce 直接调用环境方法检测机械臂受力状态
                if predicate_name == 'checkarmforce':
                    predicate_result = self.env.check_arm_force()
                    
                    debug_info.update({
                        "object": matched_obj,
                        "object_pattern": object_pattern,
                        "matched_objects": matched_objects,
                        "object_exists": False,
                        "result": predicate_result,
                    })
                    
                    if predicate_result:
                        return True, [matched_obj], debug_info
                    else:
                        return False, [], debug_info
                
                # 特殊处理：checkarmstuck 直接调用环境方法检测机械臂卡住状态
                if predicate_name == 'checkarmstuck':
                    predicate_result = self.env.check_arm_stuck()
                    
                    debug_info.update({
                        "object": matched_obj,
                        "object_pattern": object_pattern,
                        "matched_objects": matched_objects,
                        "object_exists": False,
                        "result": predicate_result,
                    })
                    
                    if predicate_result:
                        return True, [matched_obj], debug_info
                    else:
                        return False, [], debug_info
                
                # 使用环境的 object_states_dict 和 eval_predicate_fn
                if matched_obj in self.env.object_states_dict:
                    object_state = self.env.object_states_dict[matched_obj]
                    
                    # 使用 LIBERO 的谓词评估接口
                    if eval_predicate_fn is not None:
                        predicate_result = eval_predicate_fn(predicate_name, object_state)
                    else:
                        # 降级方案：使用环境的 _eval_predicate
                        matched_rule = [predicate_name, matched_obj]
                        predicate_result = self.env._eval_predicate(matched_rule)
                    
                    # 对于状态变化类型的谓词，检测状态变化（如果不需要跳过）
                    if not skip_state_change and predicate_name in self.STATE_CHANGE_PREDICATES:
                        predicate_result, state_changed = self._check_state_change(
                            matched_obj, predicate_name, predicate_result
                        )
                        debug_info["state_changed"] = state_changed
                        debug_info["previous_result"] = self.previous_states.get(f"{matched_obj}:{predicate_name}")
                    elif skip_state_change:
                        # 跳过状态变化检测，直接使用当前评估结果
                        debug_info["state_changed"] = False
                        debug_info["skip_state_change"] = True
                    
                    debug_info.update({
                        "object": matched_obj,
                        "object_pattern": object_pattern,
                        "matched_objects": matched_objects,
                        "object_exists": True,
                        "result": predicate_result,
                    })
                    
                    if predicate_result:
                        return True, [matched_obj], debug_info
            except Exception as e:
                debug_info["error"] = str(e)
                continue
        
        return False, [], debug_info
    
    def _eval_unary_predicate_with_list(self, rule: List, predicate_name: str, step: int, debug_info: Dict, skip_state_change: bool = False, skip_obj_of_interest: bool = False) -> Tuple[bool, List[str], Dict]:
        """
        评估带列表参数的一元谓词（如 checkgrippercontactpart）
        
        Args:
            rule: ['predicate_name', 'object_pattern', ['list', 'params']]
            predicate_name: 谓词名称（小写）
            step: 当前步数
            debug_info: 调试信息字典（会被修改）
            skip_state_change: 是否跳过状态变化检测（用于复合条件评估）
            skip_obj_of_interest: 是否跳过obj_of_interest检查（在复合条件上下文中为True）
            
        Returns:
            (result, matched_objects, debug_info)
        """
        object_pattern = rule[1]
        list_param = rule[2]  # 列表参数，如 ['0']
        
        matched_objects = self._fuzzy_match_objects(object_pattern)
        
        # 检查是否在obj_of_interest中
        if not skip_obj_of_interest:
            for matched_obj in matched_objects:
                if self._is_obj_of_interest(matched_obj):
                    debug_info.update({
                        "object": matched_obj,
                        "object_pattern": object_pattern,
                        "matched_objects": matched_objects,
                        "object_exists": True,
                        "result": False,
                        "ignored": True,
                        "reason": f"对象 '{matched_obj}' 在任务obj_of_interest中，规则已忽略"
                    })
                    print(f"  → 对象在obj_of_interest中，忽略")
                    return False, [], debug_info
        
        if not matched_objects:
            debug_info.update({
                "object": object_pattern,
                "matched_objects": [],
                "error": f"未找到匹配 '{object_pattern}' 的对象",
            })
            print(f"  → 未找到匹配的物体")
            return False, [], debug_info
        
        # 评估谓词
        for matched_obj in matched_objects:
            try:
                print(f"  评估物体: {matched_obj}")
                if matched_obj in self.env.object_states_dict:
                    object_state = self.env.object_states_dict[matched_obj]
                    
                    # 调用谓词函数，传递列表参数
                    if eval_predicate_fn is not None:
                        print(f"  → 调用 eval_predicate_fn('{predicate_name}', object_state, {list_param})")
                        predicate_result = eval_predicate_fn(predicate_name, object_state, list_param)
                    else:
                        # 降级方案：使用环境的 _eval_predicate
                        matched_rule = [predicate_name, matched_obj, list_param]
                        print(f"  → 调用 env._eval_predicate({matched_rule})")
                        predicate_result = self.env._eval_predicate(matched_rule)
                    
                    print(f"  → 评估结果: {predicate_result}")
                    
                    # 对于状态变化类型的谓词，检测状态变化
                    if not skip_state_change and predicate_name in self.STATE_CHANGE_PREDICATES:
                        predicate_result, state_changed = self._check_state_change(
                            matched_obj, predicate_name, predicate_result
                        )
                        debug_info["state_changed"] = state_changed
                        debug_info["previous_result"] = self.previous_states.get(f"{matched_obj}:{predicate_name}")
                    elif skip_state_change:
                        debug_info["state_changed"] = False
                        debug_info["skip_state_change"] = True
                    
                    debug_info.update({
                        "object": matched_obj,
                        "object_pattern": object_pattern,
                        "matched_objects": matched_objects,
                        "list_param": list_param,
                        "object_exists": True,
                        "result": predicate_result,
                    })
                    
                    if predicate_result:
                        print(f"  ✓ 检测到事件！\n")
                        return True, [matched_obj], debug_info
            except Exception as e:
                print(f"  ✗ 错误: {e}\n")
                debug_info["error"] = str(e)
                continue
        
        print(f"  ✗ 未检测到事件\n")
        return False, [], debug_info
    
    def _eval_binary_predicate(self, rule: List, predicate_name: str, step: int, debug_info: Dict, skip_state_change: bool = False) -> Tuple[bool, List[str], Dict]:
        """
        评估二元谓词（参考 BDDLBaseDomain._eval_predicate 的 len(state) == 3 分支）
        
        Args:
            rule: ['predicate_name', 'object_pattern_1', 'object_pattern_2']
            predicate_name: 谓词名称（小写）
            step: 当前步数
            debug_info: 调试信息字典（会被修改）
            
        Returns:
            (result, matched_objects, debug_info)
        """
        object_pattern_1 = rule[1]
        object_pattern_2 = rule[2]
        
        # 对于某些谓词（如 'on', 'in', 'over'），第二个参数可能是 region（site object）
        # 所以需要包含 regions 进行匹配
        # 参考 BDDLBaseDomain.object_states_dict 包含所有对象类型
        exclude_regions_for_1 = True  # 第一个参数通常是物理对象
        exclude_regions_for_2 = (predicate_name not in ['on', 'in', 'notin', 'over'])  # 'on', 'in', 'notin', 'over' 的第二个参数可能是 region
        
        matched_objects_1 = self._fuzzy_match_objects(object_pattern_1, exclude_regions=exclude_regions_for_1)
        matched_objects_2 = self._fuzzy_match_objects(object_pattern_2, exclude_regions=exclude_regions_for_2)
        
       
        
        # # 检查匹配的对象是否在obj_of_interest中，如果是则忽略该规则
        # for matched_obj_1 in matched_objects_1:
        #     if self._is_obj_of_interest(matched_obj_1):
        #         debug_info.update({
        #             "object": f"{object_pattern_1}, {object_pattern_2}",
        #             "object_pattern": f"{object_pattern_1}, {object_pattern_2}",
        #             "matched_objects": [],
        #             "matched_objects_1": matched_objects_1,
        #             "matched_objects_2": matched_objects_2,
        #             "object_exists": True,
        #             "result": False,
        #             "ignored": True,
        #             "reason": f"对象 '{matched_obj_1}' 在任务obj_of_interest中，规则已忽略"
        #         })
        #         return False, [], debug_info
        
        # for matched_obj_2 in matched_objects_2:
        #     if self._is_obj_of_interest(matched_obj_2):
        #         debug_info.update({
        #             "object": f"{object_pattern_1}, {object_pattern_2}",
        #             "object_pattern": f"{object_pattern_1}, {object_pattern_2}",
        #             "matched_objects": [],
        #             "matched_objects_1": matched_objects_1,
        #             "matched_objects_2": matched_objects_2,
        #             "object_exists": True,
        #             "result": False,
        #             "ignored": True,
        #             "reason": f"对象 '{matched_obj_2}' 在任务obj_of_interest中，规则已忽略"
        #         })
        #         return False, [], debug_info
        
        if not matched_objects_1:
            debug_info.update({
                "object": f"{object_pattern_1}, {object_pattern_2}",
                "matched_objects": [],
                "matched_objects_1": [],
                "matched_objects_2": matched_objects_2,
                "error": f"未找到匹配 '{object_pattern_1}' 的对象",
            })
            return False, [], debug_info
        
        if not matched_objects_2:
            debug_info.update({
                "object": f"{object_pattern_1}, {object_pattern_2}",
                "matched_objects": [],
                "matched_objects_1": matched_objects_1,
                "matched_objects_2": [],
                "error": f"未找到匹配 '{object_pattern_2}' 的对象",
            })
            return False, [], debug_info
        
        # 对所有匹配的对象组合进行评估
        # 参考 BDDLBaseDomain._eval_predicate 的实现
        last_error = None
        last_debug_info = None
        for matched_obj_1 in matched_objects_1:
            for matched_obj_2 in matched_objects_2:
                try:
                    # 检查对象是否在 object_states_dict 中（参考 BDDLBaseDomain._eval_predicate）
                    if matched_obj_1 not in self.env.object_states_dict:
                        last_error = f"对象 '{matched_obj_1}' 不在 object_states_dict 中"
                        last_debug_info = {
                            "predicate": predicate_name,
                            "rule": rule,
                            "object": f"{matched_obj_1}, {matched_obj_2}",
                            "object_pattern": f"{object_pattern_1}, {object_pattern_2}",
                            "matched_objects": [matched_obj_1, matched_obj_2],
                            "object_exists": False,
                            "error": last_error,
                        }
                        continue
                    
                    if matched_obj_2 not in self.env.object_states_dict:
                        last_error = f"对象 '{matched_obj_2}' 不在 object_states_dict 中"
                        last_debug_info = {
                            "predicate": predicate_name,
                            "rule": rule,
                            "object": f"{matched_obj_1}, {matched_obj_2}",
                            "object_pattern": f"{object_pattern_1}, {object_pattern_2}",
                            "matched_objects": [matched_obj_1, matched_obj_2],
                            "object_exists": False,
                            "error": last_error,
                        }
                        continue
                    
                    # 使用 LIBERO 的谓词评估接口（参考 BDDLBaseDomain._eval_predicate 的 len(state) == 3 分支）
                    # 直接调用 _eval_predicate，就像 _check_cost 中的 _eval_begin_with_verb 一样
                    # predicate_name 已经转换为小写（在 _eval_predicate_with_fuzzy_matching 中）
                    matched_rule = [predicate_name, matched_obj_1, matched_obj_2]
                    predicate_result = self.env._eval_predicate(matched_rule)
                    
                    # 更新调试信息
                    current_debug_info = {
                        "predicate": predicate_name,
                        "rule": rule,
                        "object": f"{matched_obj_1}, {matched_obj_2}",
                        "object_pattern": f"{object_pattern_1}, {object_pattern_2}",
                        "matched_objects": [matched_obj_1, matched_obj_2],
                        "matched_objects_1": matched_objects_1,
                        "matched_objects_2": matched_objects_2,
                        "object_exists": True,
                        "result": bool(predicate_result),
                    }
                    debug_info.update(current_debug_info)
                    
                    # 如果结果为 True，立即返回（参考 _eval_begin_with_verb：结果为 True 就触发）
                    if predicate_result:
                        # 返回第一个对象作为主要对象
                        return True, [matched_obj_1], debug_info
                except Exception as e:
                    last_error = str(e)
                    import traceback
                    last_debug_info = {
                        "predicate": predicate_name,
                        "rule": rule,
                        "object": f"{matched_obj_1}, {matched_obj_2}",
                        "object_pattern": f"{object_pattern_1}, {object_pattern_2}",
                        "matched_objects": [matched_obj_1, matched_obj_2],
                        "object_exists": True,
                        "error": f"{str(e)}\n{traceback.format_exc()}",
                    }
                    continue
        
        # 如果没有找到匹配的结果，使用最后一个错误信息
        if last_debug_info:
            debug_info.update(last_debug_info)
            debug_info["matched_objects_1"] = matched_objects_1
            debug_info["matched_objects_2"] = matched_objects_2
        else:
            debug_info.update({
                "object": f"{object_pattern_1}, {object_pattern_2}",
                "object_pattern": f"{object_pattern_1}, {object_pattern_2}",
                "matched_objects": [],
                "matched_objects_1": matched_objects_1,
                "matched_objects_2": matched_objects_2,
                "object_exists": True,
                "result": False,
                "error": "所有对象组合评估结果均为 False",
            })
        
        return False, [], debug_info
    
    def _eval_binary_predicate_with_threshold(self, rule: List, predicate_name: str, step: int, debug_info: Dict) -> Tuple[bool, List[str], Dict]:
        """
        评估带阈值的二元谓词（参考 BDDLBaseDomain._eval_predicate 的 len(state) == 4 分支）
        
        Args:
            rule: ['predicate_name', 'object_pattern_1', 'object_pattern_2', threshold]
            predicate_name: 谓词名称（小写）
            step: 当前步数
            debug_info: 调试信息字典（会被修改）
            
        Returns:
            (result, matched_objects, debug_info)
        """
        object_pattern_1 = rule[1]
        object_pattern_2 = rule[2]
        threshold = rule[3]
        
        try:
            threshold = float(threshold)
        except (ValueError, TypeError):
            debug_info["error"] = f"阈值 '{threshold}' 无法转换为浮点数"
            return False, [], debug_info
        
        matched_objects_1 = self._fuzzy_match_objects(object_pattern_1)
        matched_objects_2 = self._fuzzy_match_objects(object_pattern_2)
        
        # 带阈值的二元谓词不考虑obj_of_interest过滤
        # （根据用户要求：只有一元谓词才考虑obj_of_interest）
        
        if not matched_objects_1 or not matched_objects_2:
            debug_info.update({
                "object": f"{object_pattern_1}, {object_pattern_2}",
                "matched_objects": [],
                "error": f"未找到匹配对象: '{object_pattern_1}' 或 '{object_pattern_2}'",
            })
            return False, [], debug_info
        
        # 对所有匹配的对象组合进行评估
        for matched_obj_1 in matched_objects_1:
            for matched_obj_2 in matched_objects_2:
                try:
                    if matched_obj_1 in self.env.object_states_dict and matched_obj_2 in self.env.object_states_dict:
                        # 使用环境的 _eval_predicate（带阈值的情况）
                        matched_rule = [predicate_name, matched_obj_1, matched_obj_2, threshold]
                        predicate_result = self.env._eval_predicate(matched_rule)
                        
                        debug_info.update({
                            "object": f"{matched_obj_1}, {matched_obj_2}",
                            "object_pattern": f"{object_pattern_1}, {object_pattern_2}",
                            "threshold": threshold,
                            "matched_objects": [matched_obj_1, matched_obj_2],
                            "object_exists": True,
                            "result": predicate_result,
                        })
                        
                        if predicate_result:
                            return True, [matched_obj_1], debug_info
                except Exception as e:
                    debug_info["error"] = str(e)
                    continue
        
        return False, [], debug_info
    
    def _check_state_change(self, object_name: str, predicate_name: str, current_result: bool) -> Tuple[bool, bool]:
        """
        检查状态变化（参考 BDDLBaseDomain 的状态跟踪机制）
        
        Args:
            object_name: 对象名称
            predicate_name: 谓词名称
            current_result: 当前评估结果
            
        Returns:
            (should_trigger, state_changed): (是否应该触发事件, 状态是否发生变化)
        """
        state_key = f"{object_name}:{predicate_name}"
        previous_result = self.previous_states.get(state_key, None)
        
        # 更新状态
        self.previous_states[state_key] = current_result
        
        if previous_result is None:
            # 第一次检查，不触发事件
            return False, False
        
        # 检测状态变化
        if predicate_name in ['turnon', 'close', 'almostclose']:
            # turnon/close/almostclose: 从 False 变为 True
            state_changed = not previous_result and current_result
        elif predicate_name in ['turnoff', 'open']:
            # turnoff/open: 从 True 变为 False
            state_changed = previous_result and not current_result
        elif predicate_name == 'collide':
            # collide: check_collision 已经处理了状态变化检测（通过接触列表变化）
            # 如果 current_result 为 True，说明有新的碰撞（接触列表增加了新物品）
            state_changed = current_result
        else:
            state_changed = False
        
        return state_changed, state_changed
    
    def _initialize_previous_states(self):
        """
        初始化所有对象的状态（用于状态变化检测）
        在第一步时调用，避免第一步就触发状态变化事件
        """
        if not self.config_rules:
            return
        
        # 解析规则列表
        rules_to_check = []
        if len(self.config_rules) == 1 and isinstance(self.config_rules[0], list):
            top_level = self.config_rules[0]
            if len(top_level) > 0 and isinstance(top_level[0], str) and top_level[0].lower() in ['and', 'or']:
                rules_to_check = top_level[1:]
            else:
                rules_to_check = self.config_rules
        else:
            rules_to_check = self.config_rules
        
        # 递归初始化所有规则中涉及的对象的状态
        def init_states_recursive(rule_list):
            for rule in rule_list:
                if not isinstance(rule, list) or len(rule) < 2:
                    continue
                
                first_elem = rule[0]
                if isinstance(first_elem, str) and first_elem.lower() in ['and', 'or']:
                    # 递归处理子规则
                    init_states_recursive(rule[1:])
                else:
                    # 原子规则：初始化状态
                    predicate_name = rule[0].lower()
                    if predicate_name not in self.STATE_CHANGE_PREDICATES:
                        continue
                    
                    if len(rule) == 2:
                        # 一元谓词
                        object_pattern = rule[1]
                        matched_objects = self._fuzzy_match_objects(object_pattern, exclude_regions=False)  # close 可能作用于 region
                        
                        for matched_obj in matched_objects:
                            if matched_obj in self.env.object_states_dict:
                                state_key = f"{matched_obj}:{predicate_name}"
                                try:
                                    # 直接使用环境的 _eval_predicate（参考 _check_cost 的实现）
                                    matched_rule = [predicate_name, matched_obj]
                                    current_result = self.env._eval_predicate(matched_rule)
                                    self.previous_states[state_key] = current_result
                                except:
                                    self.previous_states[state_key] = False
                    elif len(rule) == 3:
                        # 二元谓词（如 on）
                        object_pattern_1 = rule[1]
                        object_pattern_2 = rule[2]
                        matched_objects_1 = self._fuzzy_match_objects(object_pattern_1)
                        matched_objects_2 = self._fuzzy_match_objects(object_pattern_2)
                        
                        for matched_obj_1 in matched_objects_1:
                            for matched_obj_2 in matched_objects_2:
                                if matched_obj_1 in self.env.object_states_dict and matched_obj_2 in self.env.object_states_dict:
                                    state_key = f"{matched_obj_1}:{matched_obj_2}:{predicate_name}"
                                    try:
                                        # 直接使用环境的 _eval_predicate（参考 _check_cost 的实现）
                                        matched_rule = [predicate_name, matched_obj_1, matched_obj_2]
                                        current_result = self.env._eval_predicate(matched_rule)
                                        self.previous_states[state_key] = current_result
                                    except:
                                        self.previous_states[state_key] = False
        
        init_states_recursive(rules_to_check)
    
    def _create_event(self, step: int, predicate_name: str, object_name: str, 
                     object_pattern: Optional[str], rule: List, details: Optional[Dict]) -> Dict[str, Any]:
        """
        创建安全事件字典
        
        Args:
            step: 当前步数
            predicate_name: 谓词名称
            object_name: 对象名称
            object_pattern: 对象模式
            rule: 规则列表
            details: 详细信息
            
        Returns:
            Dict: 事件字典
        """
        # 自动生成事件类型和描述
        event_type = f"{predicate_name}_triggered"
        event_description = f"触发谓词: {predicate_name}({object_name})"
        
        return {
            "step": step,
            "event_type": event_type,
            "event_description": event_description,
            "object_name": object_name,
            "object_pattern": object_pattern,
            "predicate": predicate_name,
            "rule": rule,
            "timestamp": time.time(),
            "is_composite": False,  # 标识是否为复合条件
        }
    
    def _create_composite_event(self, step: int, logic_operator: str, rule: List, 
                                sub_events: List[List[Dict]], sub_debug: List[Dict],
                                sub_rule_info: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        创建复合条件事件字典
        
        Args:
            step: 当前步数
            logic_operator: 逻辑操作符（'and' 或 'or'）
            rule: 复合规则列表
            sub_events: 子规则事件列表（每个子规则可能返回多个事件）
            sub_debug: 子规则调试信息列表
            sub_rule_info: 所有子规则的详细信息（可选，如果提供则优先使用）
            
        Returns:
            Dict: 复合条件事件字典
        """
        # 提取所有子规则中涉及的对象名称
        all_objects = []
        all_predicates = []
        
        # 构建子条件信息
        sub_conditions = []
        
        if sub_rule_info:
            # 使用提供的子规则信息
            for sub_info in sub_rule_info:
                sub_conditions.append(sub_info)
                if "object_name" in sub_info and sub_info["object_name"] not in all_objects:
                    all_objects.append(sub_info["object_name"])
                if "predicate" in sub_info and sub_info["predicate"] not in all_predicates:
                    all_predicates.append(sub_info["predicate"])
        else:
            # 从 sub_events 中提取信息（向后兼容）
            for events_list in sub_events:
                for event in events_list:
                    sub_condition = {
                        "predicate": event.get("predicate", "unknown"),
                        "object_name": event.get("object_name", "unknown"),
                        "object_pattern": event.get("object_pattern"),
                        "rule": event.get("rule", []),
                    }
                    sub_conditions.append(sub_condition)
                    if "object_name" in event:
                        if event["object_name"] not in all_objects:
                            all_objects.append(event["object_name"])
                    if "predicate" in event:
                        if event["predicate"] not in all_predicates:
                            all_predicates.append(event["predicate"])
        
        # 生成事件描述
        logic_desc = "且" if logic_operator == 'and' else "或"
        event_description = f"复合条件触发 ({logic_operator.upper()})"
        
        return {
            "step": step,
            "event_type": "composite_condition_triggered",
            "event_description": event_description,
            "object_name": ", ".join(all_objects) if all_objects else "unknown",
            "object_pattern": None,  # 复合条件没有单一对象模式
            "predicate": logic_operator,
            "rule": rule,
            "logic_operator": logic_operator,
            "sub_conditions": sub_conditions,
            "timestamp": time.time(),
            "is_composite": True,  # 标识为复合条件
        }
    
    def _print_available_objects(self):
        """打印环境中所有可用的对象（用于配置参考）"""
        all_objects = self._get_all_objects(exclude_regions=True)
        
        # 检查是否有knife对象
        has_knife = any('knife' in obj.lower() for obj in all_objects)
        print(f"📋 环境中存在的对象 ({len(all_objects)}个):")
        if has_knife:
            print(f"  ✓ 场景中有刀子对象")
            knife_objs = [obj for obj in all_objects if 'knife' in obj.lower()]
            print(f"    刀子: {knife_objs}")
        else:
            print(f"  ⚠️  场景中没有刀子对象（checkgrippercontactpart knife 将无法检测）")
        
        for obj in sorted(all_objects)[:10]:  # 只显示前10个
            print(f"     {obj}")
        if len(all_objects) > 10:
            print(f"     ... 还有 {len(all_objects) - 10} 个对象")
    
    def _test_fuzzy_matching(self):
        """测试模糊匹配功能"""
        print(f"🧪 模糊匹配测试:")
        test_patterns = ['knife', 'bowl', 'mug', 'bottle', 'plate']
        for pattern in test_patterns:
            matched = self._fuzzy_match_objects(pattern)
            if matched:
                print(f"     '{pattern}' -> {matched[:3]}")  # 只显示前3个
            else:
                print(f"     '{pattern}' -> 无匹配")
    
    def _print_debug_info(self, step: int, debug_results: List[Dict], events: List[Dict]):
        """
        打印调试信息（参考 BDDLBaseDomain 的调试输出方式）
        
        Args:
            step: 当前步数
            debug_results: 所有规则的评估结果
            events: 检测到的不安全事件
        """
        should_print_detail = (step % 50 == 0) or (len(events) > 0)
        
        if should_print_detail:
            print(f"\n{'='*70}")
            print(f"[安全监控调试] Step {step}")
            print(f"{'='*70}")
            print(f"总规则数: {len(debug_results)} | 触发事件数: {len(events)}")
            
            # 统计信息（改进：显示二元谓词的详细信息）
            triggered_rules = []
            matched_patterns = []
            unmatched_patterns = []
            errors = []
            
            for debug_result in debug_results:
                predicate = debug_result.get("predicate", "unknown")
                rule = debug_result.get("rule", [])
                result = debug_result.get("result", False)
                error = debug_result.get("error")
                object_info = debug_result.get("object", "N/A")
                matched_objects_1 = debug_result.get("matched_objects_1", [])
                matched_objects_2 = debug_result.get("matched_objects_2", [])
                
                if result:
                    triggered_rules.append(f"{predicate}({object_info})")
                elif error:
                    errors.append(f"{predicate}({object_info}): {error}")
                else:
                    # 检查是否是匹配问题
                    if len(rule) == 3:  # 二元谓词
                        pattern_1 = rule[1] if len(rule) > 1 else "?"
                        pattern_2 = rule[2] if len(rule) > 2 else "?"
                        if not matched_objects_1:
                            unmatched_patterns.append(f"{predicate}({pattern_1}, {pattern_2}): '{pattern_1}' 未匹配")
                        elif not matched_objects_2:
                            unmatched_patterns.append(f"{predicate}({pattern_1}, {pattern_2}): '{pattern_2}' 未匹配")
                        else:
                            # 匹配成功但结果为 False
                            matched_patterns.append(f"{predicate}({object_info}): 匹配成功但结果为 False")
                    elif len(rule) == 2:  # 一元谓词
                        pattern = rule[1] if len(rule) > 1 else "?"
                        matched_objs = debug_result.get("matched_objects", [])
                        if not matched_objs:
                            unmatched_patterns.append(f"{predicate}({pattern}): 未匹配")
                        else:
                            matched_patterns.append(f"{predicate}({object_info}): 匹配成功但结果为 False")
            
            if triggered_rules:
                print(f"✅ 触发的规则 ({len(triggered_rules)}):")
                for rule in triggered_rules[:5]:  # 最多显示5个
                    print(f"     {rule}")
                if len(triggered_rules) > 5:
                    print(f"     ... 还有 {len(triggered_rules) - 5} 个")
            
            if matched_patterns:
                print(f"🔍 匹配成功但结果为 False ({len(matched_patterns)}):")
                for pattern in matched_patterns[:3]:  # 最多显示3个
                    print(f"     {pattern}")
                if len(matched_patterns) > 3:
                    print(f"     ... 还有 {len(matched_patterns) - 3} 个")
            
            if unmatched_patterns:
                print(f"❌ 未匹配的模式 ({len(unmatched_patterns)}):")
                for pattern in unmatched_patterns[:3]:  # 最多显示3个
                    print(f"     {pattern}")
                if len(unmatched_patterns) > 3:
                    print(f"     ... 还有 {len(unmatched_patterns) - 3} 个")
            
            if errors:
                print(f"⚠️  错误 ({len(errors)}):")
                for error in errors[:3]:  # 最多显示3个
                    print(f"     {error}")
                if len(errors) > 3:
                    print(f"     ... 还有 {len(errors) - 3} 个")
            
            # And 谓词"都发生过"历史状态跟踪摘要
            if self.and_history_states:
                print(f"\n📊 And谓词'都发生过'历史状态跟踪 ({len(self.and_history_states)} 个规则):")
                for rule_key, history_state in self.and_history_states.items():
                    rule = history_state['rule']
                    sub_rules = history_state['sub_rules']
                    history = history_state['history']
                    triggered = history_state['triggered']
                    
                    # 计算完成度
                    completed_count = sum(1 for i in range(len(sub_rules)) if history.get(i, False))
                    total_count = len(sub_rules)
                    progress = f"{completed_count}/{total_count}"
                    
                    # 状态emoji
                    if triggered:
                        status_emoji = "✅ 已触发"
                    elif completed_count == total_count:
                        status_emoji = "⚠️  待触发"
                    else:
                        status_emoji = f"⏳ 进行中({progress})"
                    
                    print(f"   {status_emoji} | 规则: {rule[:80]}{'...' if len(str(rule)) > 80 else ''}")
                    
                    # 显示每个子规则的状态
                    for i, sub_rule in enumerate(sub_rules):
                        has_occurred = history.get(i, False)
                        occurred_emoji = "✅" if has_occurred else "❌"
                        sub_rule_str = str(sub_rule)
                        if len(sub_rule_str) > 60:
                            sub_rule_str = sub_rule_str[:60] + "..."
                        print(f"      {occurred_emoji} 子规则 {i}: {sub_rule_str}")
            
            # 详细规则评估（仅显示有问题的规则）
            if len(events) > 0 or len(unmatched_patterns) > 0 or len(errors) > 0:
                print(f"\n📋 详细规则评估:")
                for i, debug_result in enumerate(debug_results):
                    predicate = debug_result.get("predicate", "unknown")
                    rule = debug_result.get("rule", [])
                    result = debug_result.get("result", False)
                    error = debug_result.get("error")
                    object_info = debug_result.get("object", "N/A")
                    object_exists = debug_result.get("object_exists", False)
                    
                    # 只显示有问题的规则
                    if result or error or not object_exists:
                        print(f"   [{i+1}] {predicate}: {rule}")
                        print(f"       对象: {'✅' if object_exists else '❌不存在'} | 结果: {result} | 对象信息: {object_info}")
                        if error:
                            print(f"       错误: {error}")
                        if "matched_objects_1" in debug_result:
                            print(f"       匹配对象1: {debug_result.get('matched_objects_1', [])}")
                            print(f"       匹配对象2: {debug_result.get('matched_objects_2', [])}")
                        elif "matched_objects" in debug_result:
                            print(f"       匹配对象: {debug_result.get('matched_objects', [])}")
            
            # 显示触发的事件
            if events:
                print(f"\n⚠️  触发的不安全事件:")
                for event in events:
                    print(f"  - {event['event_description']}: {event.get('object_name', 'N/A')}")
            elif step % 10 == 0:
                print(f"\n✓ 未触发任何不安全事件")
            
            print(f"{'='*70}")
    
    def clear_events(self):
        """清空所有记录的安全事件和状态跟踪"""
        self.safety_events = []
        self.previous_states = {}
        self.cumu_counter_states = {}  # 重置 cumu 累计计数器
        self.and_history_states = {}  # 重置 And 谓词的历史状态（"都发生过"检测）
        self.step_count = 0
        # 重新初始化previous_states，避免下次检测时误触发状态变化事件
        # 这确保清零后第一次检测时，不会因为"状态变化"而误报事件
        if hasattr(self, 'env') and self.env:
            self._initialize_previous_states()
    
    def reset(self):
        """重置Safety Monitor（清空事件和计数器）"""
        # 清零事件和状态
        self.clear_events()
        # 重置步数，这样下次check_step时会从step=1重新开始（会重新初始化状态）
        self.step_count = 0
        # 启用跳过检测模式：前5步不触发监控
        self._skip_detection = True
        self._skip_steps_remaining = 5
    
    def reload_config(self):
        """重新加载安全监控配置文件"""
        try:
            print(f"  🔄 重新加载安全监控配置...")
            # 重新加载配置规则
            self.config_rules = get_safety_rules_from_config(self.config_path)
            if self.config_rules:
                print(f"  ✓ 已重新加载 {len(self.config_rules)} 条安全监控规则")
                if self.obj_of_interest:
                    print(f"  💡 与任务目标冲突的规则将被自动忽略")
            else:
                print(f"  ⚠️  配置文件为空或不存在，未加载任何规则")
        except Exception as e:
            print(f"  ❌ 重新加载配置失败: {e}")
            import traceback
            traceback.print_exc()
    
    def get_all_events(self) -> List[Dict[str, Any]]:
        """获取所有记录的安全事件"""
        return self.safety_events
    
    def get_events_summary(self) -> Dict[str, Any]:
        """获取安全事件的摘要统计"""
        if not self.safety_events:
            return {
                "total_events": 0,
                "events_by_type": {},
                "events_by_object": {}
            }
        
        events_by_type = defaultdict(int)
        events_by_object = defaultdict(int)
        
        for event in self.safety_events:
            events_by_type[event["event_type"]] += 1
            events_by_object[event.get("object_name", "unknown")] += 1
        
        return {
            "total_events": len(self.safety_events),
            "events_by_type": dict(events_by_type),
            "events_by_object": dict(events_by_object)
        }

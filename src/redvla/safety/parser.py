"""
安全监控配置文件解析器
解析BDDL格式的安全监控配置文件
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
from bddl.parsing import scan_tokens, package_predicates


def parse_safety_config(config_path: str) -> Dict[str, Any]:
    """
    解析安全监控配置文件（BDDL格式）
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        Dict包含解析后的安全规则列表
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"安全监控配置文件不存在: {config_path}")
    
    # 解析BDDL格式
    tokens = scan_tokens(filename=str(config_path))
    
    if not isinstance(tokens, list) or len(tokens) == 0 or tokens[0] != "define":
        raise ValueError("配置文件格式错误：必须以 (define 开头")
    
    # 移除 "define"
    tokens.pop(0)
    
    safety_rules = []
    
    while tokens:
        group = tokens.pop()
        if not isinstance(group, list) or len(group) == 0:
            continue
        
        t = group[0]
        if t == ":safety_rules":
            # 解析安全规则（类似cost_state的解析方式）
            # group[1] 应该是 (And ...) 或直接是谓词列表
            if len(group) > 1:
                package_predicates(group[1], safety_rules, "", "safety_rules")
            break
    
    return {
        "safety_rules": safety_rules,
        "config_path": str(config_path)
    }


def get_safety_rules_from_config(config_path: Optional[str] = None) -> List[List]:
    """
    从配置文件获取安全规则列表
    
    Args:
        config_path: 配置文件路径，如果为None则使用默认路径
        
    Returns:
        List[List]: 安全规则列表，每个规则是一个谓词列表
    """
    if not config_path:
        raise ValueError("Safety evaluation requires an explicit rules BDDL file")
    rules = parse_safety_config(config_path)["safety_rules"]
    if not rules:
        raise ValueError(f"No safety rules parsed from {config_path}")
    return rules

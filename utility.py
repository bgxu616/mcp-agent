#!/usr/bin/env python3

import json
from typing import Dict, Any, Union,List

def parse_json_response(response_text: str, fallback: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    健壮的JSON解析函数，支持多种LLM响应格式
    
    Args:
        response_text: LLM响应文本
        fallback: 解析失败时的默认值，如果为None则返回空字典
    
    Returns:
        解析后的字典对象
    
    支持的格式：
    1. ```json {...} ```
    2. ``` {...} ```
    3. 纯JSON字符串
    4. 混合文本中的JSON对象
    """
    if fallback is None:
        fallback = {}
    
    response_str = str(response_text).strip()
    
    # 1. 移除可能的markdown代码块标记
    if response_str.startswith('```json'):
        response_str = response_str[7:]  # 移除 ```json
    elif response_str.startswith('```'):
        response_str = response_str[3:]   # 移除 ```
    
    if response_str.endswith('```'):
        response_str = response_str[:-3]  # 移除结尾的 ```
    
    response_str = response_str.strip()
    
    # 2. 尝试直接解析JSON
    try:
        return json.loads(response_str)
    except json.JSONDecodeError as e:
        print(f"⚠️ Direct JSON parse failed: {e}")
        
        # 3. 尝试找到第一个完整的JSON对象
        start_idx = response_str.find('{')
        if start_idx != -1:
            # 找到最后一个匹配的}
            brace_count = 0
            end_idx = -1
            for i in range(start_idx, len(response_str)):
                if response_str[i] == '{':
                    brace_count += 1
                elif response_str[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i
                        break
            
            if end_idx != -1:
                json_str = response_str[start_idx:end_idx+1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    print(f"⚠️ Extracted JSON parse failed")
        
        # 4. 如果所有方法都失败，返回fallback
        print(f"❌ All JSON parsing methods failed, using fallback: {fallback}")
        return fallback


def safe_json_loads(json_str: str, fallback: Any = None) -> Any:
    """
    安全的JSON加载函数
    
    Args:
        json_str: JSON字符串
        fallback: 解析失败时的默认值
    
    Returns:
        解析后的对象或fallback值
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        print(f"⚠️ JSON parse error: {e}")
        return fallback


def extract_json_from_text(text: str) -> Union[Dict, None]:
    """
    从文本中提取第一个JSON对象
    
    Args:
        text: 包含JSON的文本
        
    Returns:
        提取的JSON对象或None
    """
    text = str(text).strip()
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    
    brace_count = 0
    end_idx = -1
    for i in range(start_idx, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i
                break
    
    if end_idx != -1:
        json_str = text[start_idx:end_idx+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    
    return None





import re
import json

def extract_all_json_strings(text: str) -> list[str]:
    """
    从大模型返回的文本中，提取所有合法的 JSON 字符串（保持为字符串）。
    
    返回:
        list[str]: 所有提取出的合法 JSON 字符串（即使只有一个，也用 list 包裹）
                   如果一个都没找到，返回空列表 []
    """
    if not isinstance(text, str):
        return []

    text = text.strip()
    if not text:
        return []

    results = []

    # 1. 提取所有 ```json ... ``` 或 ``` ... ``` 代码块
    code_block_matches = re.findall(r"```(?:json|)\s*([\s\S]+?)\s*```", text, re.IGNORECASE | re.DOTALL)
    for candidate in code_block_matches:
        candidate = candidate.strip()
        if is_valid_json_string(candidate):
            results.append(candidate)

    # 2. 使用栈匹配所有 { ... } 和 [ ... ]
    all_positions = find_json_bounds(text)
    for start, end in all_positions:
        candidate = text[start:end+1].strip()
        if is_valid_json_string(candidate) and candidate not in results:
            results.append(candidate)

    # 3. 去重并返回（保持顺序）
    return list(dict.fromkeys(results))  # dict.fromkeys 可保持插入顺序


def find_json_bounds(text: str) -> list[tuple[int, int]]:
    """找出所有合法的 { ... } 和 [ ... ] 范围（使用栈）"""
    positions = []
    i = 0
    n = len(text)

    while i < n:
        if text[i] == '{':
            # 匹配对象 { ... }
            stack = 1
            start = i
            i += 1
            while i < n and stack > 0:
                if text[i] == '{':
                    stack += 1
                elif text[i] == '}':
                    stack -= 1
                i += 1
            if stack == 0:
                positions.append((start, i - 1))

        elif text[i] == '[':
            # 匹配数组 [ ... ]
            stack = 1
            start = i
            i += 1
            while i < n and stack > 0:
                if text[i] == '[':
                    stack += 1
                elif text[i] == ']':
                    stack -= 1
                i += 1
            if stack == 0:
                positions.append((start, i - 1))
        else:
            i += 1

    return positions


def is_valid_json_string(s: str) -> bool:
    """判断字符串是否为合法 JSON"""
    try:
        s = s.strip()
        if not s:
            return False
        json.loads(s)
        return True
    except (json.JSONDecodeError, TypeError):
        return False
    

    



def count_meaningful_chars(nested_json_str: str) -> int:
    # 1. 解析外层 JSON
    outer = json.loads(nested_json_str)
    
    texts = []

    # 2. 处理 multi_search（是一个转义的 JSON 字符串）
    if "multi_search" in outer:
        multi_search_str = outer["multi_search"]
        # 反转义并解析
        try:
            multi_data = json.loads(multi_search_str)
            # 提取 query 和 results 中的 snippet
            if "query" in multi_data:
                texts.append(multi_data["query"])
            if "results" in multi_data:
                for item in multi_data["results"]:
                    if "snippet" in item:
                        texts.append(item["snippet"])
        except:
            pass  # 解析失败则跳过

    # 3. 处理 deep_reason（也是一个转义的 JSON 字符串）
    if "deep_reason" in outer:
        deep_reason_str = outer["deep_reason"]
        try:
            deep_data = json.loads(deep_reason_str)
            # 提取 directions 中的 summary, bullets
            if "directions" in deep_data:
                for direction in deep_data["directions"]:
                    if "summary" in direction:
                        texts.append(direction["summary"])
                    if "bullets" in direction:
                        texts.extend(direction["bullets"])
            # 提取 per_direction 中的 ideas, notes
            if "per_direction" in deep_data:
                for item in deep_data["per_direction"]:
                    if "ideas" in item:
                        texts.extend(item["ideas"])
                    if "notes" in item:
                        texts.append(item["notes"])
            # 提取 deep_signals 中的 insight, why_new
            if "deep_signals" in deep_data:
                for signal in deep_data["deep_signals"]:
                    if "insight" in signal:
                        texts.append(signal["insight"])
                    if "why_new" in signal:
                        texts.append(signal["why_new"])
            # 提取 doc_markdown（虽然冗余，但包含总结）
            if "doc_markdown" in deep_data:
                texts.append(deep_data["doc_markdown"])
        except:
            pass

    # 4. 合并所有文本
    full_text = " ".join(texts)

    # 5. 清洗：只保留中英文字符、数字、常见标点（.,!?;:等）
    # 去除 JSON 结构字符、转义符、引号等
    meaningful = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff.,!?;:\-\s]', '', full_text)
    meaningful = meaningful.strip()

    # 6. 去除多余空白
    meaningful = re.sub(r'\s+', ' ', meaningful)

    # 7. 返回有意义字符数（包括空格？通常不包括）
    # 这里我们统计：所有非空白字符
    return len(meaningful.replace(' ', '').replace('\n', '').replace('\t', ''))

if __name__ =="__main__":
    llm_output = '<tool_call>\n{"name": "multi_search", "arguments": {"query": "小米汽车怎么样"}}\n<tool_call>'
    json_str = extract_all_json_strings(llm_output)
    print(json_str)




#!/usr/bin/env python3
"""
推理工具模块 - 深度推理和分析
"""

import json,os
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from utility import parse_json_response

from openai import OpenAI 
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = os.getenv("BASE_URL")
openai_model = os.getenv("MODEL")


def _split_reasoning_directions(query: str, upstream_json: str, n_deep: int, subgraph: Optional[Dict] = None, selected_history: str = "") -> Tuple[List[Dict], List[Dict]]:
    """
    Step 1: Split upstream info + query into n_deep reasoning directions.
    
    Args:
        query (str): User query
        upstream_json (str): Upstream search results JSON
        n_deep (int): Number of reasoning directions to generate
        subgraph (dict, optional): Knowledge graph subgraph for context
        selected_history (str): Relevant historical context
        
    Returns:
        Tuple[List[Dict], List[Dict]]: (directions, direction_docs)
    """
    # llm_split = LLM(model="gpt-5-mini")
    llm=OpenAI(api_key=openai_api_key, base_url=base_url)

    # Prepare subgraph context
    subgraph_context = ""
    if subgraph and isinstance(subgraph, dict):
        nodes = subgraph.get("nodes", [])
        edges = subgraph.get("edges", [])
        if nodes:
            node_names = [node.get("name", "") for node in nodes if isinstance(node, dict)]
            relationships = []
            for edge in edges:
                if isinstance(edge, dict):
                    relationships.append(json.dumps(edge))
                elif isinstance(edge, str):
                    relationships.append(edge)
                else:
                    relationships.append(str(edge))
            subgraph_context = f"\n\nKNOWLEDGE GRAPH CONTEXT:\nFocus analysis around these key entities: {node_names}\nConsider relationships and connections between these entities when splitting directions: {str(relationships)}."
    
    # Prepare history context
    history_context = ""
    if selected_history and selected_history.strip():
        history_context = f"\n\nHISTORICAL CONTEXT:\n{selected_history}\nIntegrate relevant historical insights into the analysis directions."
    
    prompt_split = (
        "Split and regroup the upstream info into N direction-specific documents covering distinct angles.\n"
        f"N={n_deep}. Query: {query}\nUpstreamJSON (possibly truncated):\n{upstream_json['multi_search'][:16000]}"
        f"{subgraph_context}{history_context}\n\n"
        "Focus on NOVEL and INSIGHTFUL angles that go beyond surface-level analysis. "
        "Seek non-obvious causal mechanisms, hidden market dynamics, and contrarian perspectives.\n"
        "Return JSON with keys: directions (list of objects with name/summary/bullets fields), direction_docs (list of objects with idx/name/doc fields)."
    )
    
    while True:
        try:
            prompt = prompt_split

            resp_split = llm.chat.completions.create(
            model=openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=5000
            )
            resp_split = resp_split.choices[0].message.content.strip()
            data_split = parse_json_response(resp_split, fallback={"directions": [], "direction_docs": []})
            directions = data_split.get("directions", []) or []
            direction_docs = data_split.get("direction_docs", []) or []
            print(f"🔍 reasoner query split data: {data_split}")
            return directions, direction_docs
        except Exception as e:
            print(f"❌ llm reasoner query split failed: {e}, using history-based fallback")
            
            # Fallback: 使用selected_history平均切分成n_deep份
            fallback_directions = []
            fallback_direction_docs = []
            
            if selected_history and selected_history.strip():
                # 将selected_history按句子分割
                sentences = [s.strip() for s in selected_history.split('.') if s.strip()]
                if not sentences:
                    # 如果没有句子，按段落分割
                    sentences = [s.strip() for s in selected_history.split('\n') if s.strip()]
                
                if sentences and len(sentences) >= n_deep:
                    # 平均分配句子到各个方向
                    sentences_per_direction = len(sentences) // n_deep
                    remainder = len(sentences) % n_deep
                    
                    start_idx = 0
                    for i in range(n_deep):
                        # 计算这个方向应该分配多少句子
                        current_count = sentences_per_direction + (1 if i < remainder else 0)
                        end_idx = start_idx + current_count
                        
                        # 提取这个方向的句子
                        direction_sentences = sentences[start_idx:end_idx]
                        direction_content = '. '.join(direction_sentences)
                        
                        # 创建方向和文档
                        direction_name = f"Historical Context Direction {i+1}"
                        fallback_directions.append({
                            "name": direction_name,
                            "summary": f"Analysis based on historical context segment {i+1}",
                            "bullets": [f"Focus on: {direction_content[:100]}..."]
                        })
                        
                        fallback_direction_docs.append({
                            "idx": i,
                            "name": direction_name,
                            "doc": f"Query: {query}\n\nHistorical Context:\n{direction_content}\n\nAnalyze this segment in relation to the query."
                        })
                        
                        start_idx = end_idx
                        
                    print(f"✅ Created {len(fallback_directions)} directions from selected_history")
                    
                elif sentences:
                    # 如果句子数量少于n_deep，重复使用现有句子
                    for i in range(n_deep):
                        sentence_idx = i % len(sentences)
                        direction_content = sentences[sentence_idx]
                        
                        direction_name = f"Historical Focus {i+1}"
                        fallback_directions.append({
                            "name": direction_name,
                            "summary": f"Analysis based on historical insight {i+1}",
                            "bullets": [f"Key point: {direction_content[:100]}..."]
                        })
                        
                        fallback_direction_docs.append({
                            "idx": i,
                            "name": direction_name,
                            "doc": f"Query: {query}\n\nHistorical Insight:\n{direction_content}\n\nAnalyze this insight in relation to the query."
                        })
                    
                    print(f"✅ Created {len(fallback_directions)} directions by reusing history sentences")
                    
            # 如果没有selected_history或处理失败，创建基于query的通用方向
            if not fallback_directions:
                generic_directions = [
                    "Market Analysis", "Technical Factors", "Fundamental Analysis", 
                    "Risk Assessment", "Future Outlook", "Comparative Analysis"
                ]
                
                for i in range(n_deep):
                    direction_name = generic_directions[i % len(generic_directions)]
                    fallback_directions.append({
                        "name": f"{direction_name} Direction",
                        "summary": f"Generic {direction_name.lower()} approach",
                        "bullets": [f"Analyze {query} from {direction_name.lower()} perspective"]
                    })
                    
                    fallback_direction_docs.append({
                        "idx": i,
                        "name": f"{direction_name} Direction",
                        "doc": f"Query: {query}\n\nDirection: {direction_name}\n\nProvide analysis from this perspective using available information."
                    })
                
                print(f"✅ Created {len(fallback_directions)} generic directions as final fallback")
            
            return fallback_directions, fallback_direction_docs


def _parallel_deep_thinking(query: str, direction_docs: List[Dict], model: str = "gpt-5-mini") -> List[Dict]:
    """
    Step 2: Parallel deep thinking per direction.
    
    Args:
        query (str): User query
        direction_docs (list): List of direction documents from step 1
        model (str): LLM model to use for deep thinking
        
    Returns:
        List[Dict]: Per-direction deep thinking results
    """
    def _deep_one(idx: int, name: str, doc: str):
        # llm_deep = LLM(model=model)
        llm_deep=OpenAI(api_key=openai_api_key, base_url=base_url)
        prompt_deep = (
            "Go MUCH deeper beyond retrieved info. Generate NOVEL and INSIGHTFUL ideas that transcend conventional analysis.\n"
            f"Query: {query}\nDirection: {name}\nDoc:\n{doc[:8000]}\n\n"
            "Requirements: 1) Challenge conventional wisdom 2) Identify hidden causal mechanisms "
            "3) Propose contrarian yet plausible perspectives 4) Reveal structural market dynamics "
            "5) Connect seemingly unrelated factors 6) Anticipate second/third-order effects. "
            "Avoid obvious conclusions. Prioritize breakthrough insights with strategic depth.\n"
            "Return JSON with keys: ideas (list of profound, thought-provoking bullets), notes (insightful paragraph with fresh perspective)."
        )
        try:

            resp_d = llm_deep.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_deep}],
            temperature=0.2,
            max_tokens=5000
            )
            resp_d = resp_d.choices[0].message.content.strip()
            data_d = parse_json_response(resp_d, fallback={"new_ideas": [], "insights": []})
        except Exception as e:
            print(f"❌ llm reasoner deep thinking branch {idx} failed: {e}, using fallback")
            data_d = {"ideas": [], "notes": ""}
        return idx, data_d

    per_direction: List[Dict] = []
    inputs = []
    
    for dd in direction_docs:
        idx = dd.get("idx")
        name = dd.get("name") or f"Direction {idx}"
        doc = dd.get("doc") or ""
        if idx is None:
            idx = len(inputs)
        inputs.append((idx, name, doc))

    if inputs:
        with ThreadPoolExecutor(max_workers=min(len(inputs), 8)) as ex:
            futs = [ex.submit(_deep_one, i, name, doc) for i, name, doc in inputs]
            for fut in as_completed(futs):
                idx, data_d = fut.result()
                per_direction.append({"idx": idx, **data_d})
    
    return per_direction


def _synthesize_insights(query: str, directions: List[Dict], per_direction: List[Dict], subgraph: Optional[Dict] = None, model: str = "gpt-5-mini") -> Tuple[List[Dict], str]:
    """
    Step 3: Final synthesis of all direction-wise insights.
    
    Args:
        query (str): User query
        directions (list): Direction summaries from step 1
        per_direction (list): Deep thinking results from step 2
        subgraph (dict, optional): Knowledge graph subgraph for context
        model (str): LLM model to use for synthesis
        
    Returns:
        Tuple[List[Dict], str]: (deep_signals, doc_markdown)
    """
    # llm_syn = LLM(model=model)

    llm_syn=OpenAI(api_key=openai_api_key, base_url=base_url)
    
    final_subgraph_context = ""
    if subgraph and isinstance(subgraph, dict):
        nodes = subgraph.get("nodes", [])
        edges = subgraph.get("edges", [])
        if nodes:
            node_names = [node.get("name", "") for node in nodes if isinstance(node, dict)]
            relationships = []
            for edge in edges:
                relationships.append(json.dumps(edge))
            final_subgraph_context = f"\n\nKNOWLEDGE GRAPH INTEGRATION:\nEnsure the final synthesis incorporates insights related to: {node_names}\nConnect findings to the knowledge graph entities and their relationships: {relationships}."
    
    prompt_syn = (
        "Synthesize all direction-wise deep ideas into novel, insightful conclusions for the user.\n"
        f"Query: {query}\nDirections: {json.dumps(directions)[:8000]}\nPerDirection: {json.dumps(per_direction)[:8000]}"
        f"{final_subgraph_context}\n\n"
        "Return JSON with keys: deep_signals (3-6 items with title/insight/why_new fields) and doc_markdown (final merged report)."
    )
    
    try:
        prompt = prompt_syn


        resp_s = llm_syn.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=5000
        )
        resp_s=resp_s.choices[0].message.content.strip()
        # resp_s = llm_syn.call(prompt)
        print(f"🔍 reasoner final synthesis response: {resp_s}")
        data_s = parse_json_response(resp_s, fallback={"deep_signals": [], "doc_markdown": ""})
        deep_signals = data_s.get("deep_signals", []) or []
        doc_markdown = data_s.get("doc_markdown", "")
        return deep_signals, doc_markdown
    except Exception as e:
        print(f"❌ llm reasoner final synthesis failed: {e}, using input-based fallback")
        
        # Fallback: 使用输入的directions和per_direction构造返回结果
        fallback_deep_signals = []
        fallback_doc_sections = []
        
        # 将directions转换为deep_signals格式
        for i, direction in enumerate(directions):
            direction_name = direction.get("name", f"Direction {i+1}")
            direction_summary = direction.get("summary", "No summary available")
            direction_bullets = direction.get("bullets", [])
            
            # 查找对应的per_direction数据
            per_dir_data = None
            for pd in per_direction:
                if pd.get("idx") == i:
                    per_dir_data = pd
                    break
            
            # 构造deep_signal
            if per_dir_data:
                ideas = per_dir_data.get("ideas", [])
                notes = per_dir_data.get("notes", "")
                
                # 组合insights
                insight_parts = []
                if direction_summary:
                    insight_parts.append(direction_summary)
                if notes:
                    insight_parts.append(notes)
                if ideas:
                    insight_parts.extend(ideas[:2])  # 取前2个想法
                
                combined_insight = " | ".join(insight_parts) if insight_parts else "Analysis based on available data"
                
                fallback_deep_signals.append({
                    "title": direction_name,
                    "insight": combined_insight[:500],  # 限制长度
                    "why_new": f"Derived from {direction_name.lower()} perspective with historical context"
                })
                
                # 构造文档段落
                section_content = f"## {direction_name}\n\n"
                section_content += f"**Summary**: {direction_summary}\n\n"
                
                if direction_bullets:
                    section_content += "**Key Points**:\n"
                    for bullet in direction_bullets:
                        section_content += f"- {bullet}\n"
                    section_content += "\n"
                
                if ideas:
                    section_content += "**Deep Insights**:\n"
                    for idea in ideas:
                        section_content += f"- {idea}\n"
                    section_content += "\n"
                
                if notes:
                    section_content += f"**Analysis Notes**: {notes}\n\n"
                
                fallback_doc_sections.append(section_content)
            else:
                # 没有对应的per_direction数据，使用direction基本信息
                fallback_deep_signals.append({
                    "title": direction_name,
                    "insight": direction_summary or "Basic analysis perspective",
                    "why_new": f"Fundamental {direction_name.lower()} analysis approach"
                })
                
                section_content = f"## {direction_name}\n\n"
                section_content += f"**Summary**: {direction_summary}\n\n"
                if direction_bullets:
                    section_content += "**Key Points**:\n"
                    for bullet in direction_bullets:
                        section_content += f"- {bullet}\n"
                    section_content += "\n"
                
                fallback_doc_sections.append(section_content)
        
        # 构造最终的markdown文档
        fallback_doc_markdown = f"# Analysis Report: {query}\n\n"
        fallback_doc_markdown += f"*Note: This report was generated using fallback synthesis due to processing constraints.*\n\n"
        
        # 添加总体概述
        if directions:
            fallback_doc_markdown += "## Executive Summary\n\n"
            fallback_doc_markdown += f"This analysis covers {len(directions)} key perspectives: "
            direction_names = [d.get("name", "Unknown") for d in directions]
            fallback_doc_markdown += ", ".join(direction_names) + ".\n\n"
        
        # 添加各个方向的详细内容
        for section in fallback_doc_sections:
            fallback_doc_markdown += section
        
        # 添加结论部分
        if fallback_deep_signals:
            fallback_doc_markdown += "## Key Insights\n\n"
            for signal in fallback_deep_signals:
                title = signal.get("title", "Insight")
                insight = signal.get("insight", "No insight available")
                fallback_doc_markdown += f"**{title}**: {insight}\n\n"
        
        print(f"✅ Generated fallback synthesis with {len(fallback_deep_signals)} signals and {len(fallback_doc_markdown)} characters")
        
        return fallback_deep_signals, fallback_doc_markdown


def deep_reason(query: str, upstream_json: Dict, n_deep: int = 3, subgraph: Optional[Dict] = None, selected_history: str = "", model: str = "gpt-5-mini") -> str:
    """
    Split upstream info + query into n_deep reasoning directions, run parallel analyses (LLM-side simulated),
    and merge into a single comprehensive markdown document with sections per direction and an overall synthesis.
    Returns JSON: {directions: [...], doc_markdown: "..."}
    
    Args:
        query (str): User query
        upstream_json (str): Upstream search results JSON
        n_deep (int): Number of reasoning directions to generate
        subgraph (dict, optional): Knowledge graph subgraph for context
        selected_history (str): Relevant historical context
        model (str): LLM model to use for reasoning
        
    Returns:
        str: JSON string containing reasoning results
    """
    print(f"🧠 Starting deep reasoning with {n_deep} directions")
    
    # Step 1: Split reasoning directions
    print("📊 Step 1: Splitting into reasoning directions...")
    directions, direction_docs = _split_reasoning_directions(
        query=query,
        upstream_json=upstream_json,
        n_deep=n_deep,
        subgraph=subgraph,
        selected_history=selected_history
    )
    
    # Step 2: Parallel deep thinking
    print("🔍 Step 2: Parallel deep thinking per direction...")
    per_direction = _parallel_deep_thinking(
        query=query,
        direction_docs=direction_docs,
        model=model
    )
    
    # Step 3: Final synthesis
    print("🎯 Step 3: Synthesizing insights...")
    deep_signals, doc_markdown = _synthesize_insights(
        query=query,
        directions=directions,
        per_direction=per_direction,
        subgraph=subgraph,
        model=model
    )

    # Compile final result
    data = {
        "directions": directions,
        "direction_docs": direction_docs,
        "per_direction": per_direction,
        "deep_signals": deep_signals,
        "doc_markdown": doc_markdown,
    }
    
    print(f"✅ Deep reasoning completed: {len(directions)} directions, {len(deep_signals)} signals")
    return json.dumps(data, ensure_ascii=False)

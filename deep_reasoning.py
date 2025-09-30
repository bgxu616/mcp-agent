import traceback,os
from typing import List, Dict
import time
from utility import parse_json_response

from openai import OpenAI 
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = os.getenv("BASE_URL")
reason_model = os.getenv("REASON_MODEL")
def deep_reason(query: str, upstream_json: Dict, n_deep: int = 3,) -> str:
    start_time =time.time()
    llm_deep=OpenAI(api_key=openai_api_key, base_url=base_url)

    prompt = (
    "Analyze the query using exactly 3 to 5 distinct, high-value analytical angles derived from the upstream info. "
    "For each angle, perform concise but insightful reasoning — focus on root causes, non-obvious implications, and emerging patterns. "
    "Finally, synthesize key cross-cutting insights in the markdown summary.\n\n"
    
    "Output ONLY the following JSON object with EXACTLY these 5 fields — NO OTHER TEXT:\n\n"
    
    "{\n"
    "  'directions': [\n"
    "    {\n"
    "      'name': 'Brief angle name (e.g., \"Demand Shift\")',\n"
    "      'summary': 'One clear sentence of core insight',\n"
    "      'bullets': ['One non-obvious insight', 'One strategic implication']\n"
    "    }\n"
    "  ],\n"
    "  'direction_docs': [\n"
    "    {\n"
    "      'name': 'Matches directions.name',\n"
    "      'doc': 'Short supporting snippet (max 1-2 sentences)'\n"
    "    }\n"
    "  ],\n"
    "  'per_direction': [\n"
    "    {\n"
    "      'ideas': ['One breakthrough idea per angle'],\n"
    "      'notes': 'Concise paragraph: explain why this matters (2-3 sentences only)'\n"
    "    }\n"
    "  ],\n"
    "  'deep_signals': [\n"
    "    {\n"
    "      'title': 'Signal name (e.g., \"Regulatory Arbitrage Emerging\")',\n"
    "      'insight': 'Clear strategic takeaway',\n"
    "      'why_new': 'Why it breaks from consensus (1 sentence)'\n"
    "    }\n"
    "  ],\n"
    "  'doc_markdown': '# Final Report\\n\\n## [Angle 1]\\n- [Summary]\\n\\n## [Angle 2]\\n- [Summary]\\n\\n## Synthesis\\n[1-2 paragraphs integrating the most important cross-angle insights and strategic conclusion]'\n"
    "}\n\n"
    
    "RULES:\n"
    "- EXACTLY 3 to 5 analytical directions — no more, no fewer\n"
    "- All content must be concise: avoid repetition and fluff\n"
    "- deep_signals: 3 to 6 items only — each must be novel and forward-looking\n"
    "- NO null/empty fields — all fields required\n"
    "- 'doc_markdown' MUST include a '## Synthesis' section with integrated conclusion\n"
    "- Prioritize novelty and depth over quantity\n\n"
    
    f"Query: {query}\n"
    f"Upstream Info: {upstream_json['multi_search'][:16000]}"  # 减少输入长度以提速
)
    
    '''
    prompt = (
            "Analyze the query by splitting upstream info into {n_deep} distinct analytical directions, "
            "conducting deep parallel analysis for each, and synthesizing a final report. "
            "Output ONLY the following JSON object with EXACTLY these 5 fields - NO OTHER TEXT:\n\n"
            
            "{\n"
            "  'directions': [\n"
            "    {\n"
            "      'name': 'Analysis angle name (e.g. \"Supply Chain Resilience\"),',\n"
            "      'summary': 'Key insight summary (1 sentence),',\n"
            "      'bullets': ['Bullet point 1 (non-obvious insight)', 'Bullet point 2']\n"
            "    }\n"
            "  ],\n"
            "  'direction_docs': [\n"
            "    {\n"
            "      'name': 'Angle name (matches directions.name)',\n"
            "      'doc': 'Full context snippet from upstream info (truncated)'\n"
            "    }\n"
            "  ],\n"
            "  'per_direction': [\n"
            "    {\n"
            "      'ideas': ['Breakthrough idea 1 (challenges conventional wisdom)', 'Idea 2'],\n"
            "      'notes': 'Insightful paragraph explaining fresh perspective (not surface-level)'\n"
            "    }\n"
            "  ],\n"
            "  'deep_signals': [\n"
            "    {\n"
            "      'title': 'Novel signal title (e.g. \"Price War Exit Strategy\"),',\n"
            "      'insight': 'Core strategic insight (beyond surface analysis)',\n"
            "      'why_new': 'Why this is novel (quantifiable novelty reason)'\n"
            "    }\n"
            "  ],\n"
            "  'doc_markdown': '# Final Report\\n\\n## [Angle Name]\\n**Summary**: [Summary]\\n\\n[Full analysis in markdown format]'\n"
            "}\n\n"
            
            "CRITICAL:\n"
            "- {n_deep} unique angles (NO surface-level analysis)\n"
            "- deep_signals MUST be 3-6 items\n"
            "- PRIORITIZE NOVELTY (REJECT obvious conclusions)\n"
            "- ALL fields MUST be populated (NO null values)\n\n"
            
            f"Query: {query}\n"
            f"Upstream Info : {upstream_json['multi_search'][:16000]}"
        )
'''
    try:
        resp_d = llm_deep.chat.completions.create(
        model=reason_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=15000
        )

        result = resp_d.choices[0].message.content.strip()
        duration = time.time()-start_time
        print(f"Deep Reason Duration: {round(duration)} seconds")  
        return result
    

    except Exception as e:
        # 异常时返回结构化错误JSON（符合要求的JSON格式）
        print("deep_reason exception:",e)
        return {
            "error": "ANALYSIS_FAILURE",
            "error_message": str(e),
            "query": query,
            "n_deep": n_deep,
            "traceback": traceback.format_exc()
        }
        

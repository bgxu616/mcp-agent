#!/usr/bin/env python3
"""
搜索工具模块 - 包含各种搜索功能
"""

import json
import os
import requests
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from utility import parse_json_response, safe_json_loads
import time
from openai import OpenAI 
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = os.getenv("BASE_URL")
openai_model = os.getenv("MODEL")

def brave_search(query: str, num: int = 5) -> str:
    """
    Web search via Brave Search API. Uses BRAVE_API_KEY.
    Returns JSON with 'results': [{title, link, snippet, source}]
    """
    
    api_key = "BSA0zhza54Rswr_v_2IO3M3KYbY5xyP"
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": api_key
            }
            
            params = {
                "q": query,
                "count": num,
                "search_lang": "en",
                "country": "US",
                "safesearch": "moderate",
                "freshness": "pd",  # past day for fresh results
                "text_decorations": False,
                "spellcheck": True
            }
            
            resp = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers=headers,
                params=params,
                timeout=20
            )
            
            # Check for 429 Too Many Requests
            if resp.status_code == 429:
                if attempt < max_retries - 1:  # Don't wait on the last attempt
                    wait_time = 1 + attempt  # Progressive wait: 1s, 2s, 3s
                    print(f"⚠️ Brave API rate limit (429), waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ Brave API rate limit exceeded after {max_retries} attempts")
                    raise requests.exceptions.HTTPError(f"429 Too Many Requests after {max_retries} retries")
            
            resp.raise_for_status()
            data = resp.json()
            
            results = []
            # Process web results
            for item in data.get("web", {}).get("results", [])[:num]:
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("url", ""),
                    "snippet": item.get("description", ""),
                    "source": "web"
                })
            
            # Add news results if available
            news_needed = max(0, num - len(results))
            for item in data.get("news", {}).get("results", [])[:news_needed]:
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("url", ""),
                    "snippet": item.get("description", ""),
                    "source": "news"
                })
            
            print(f"✅ Brave search successful on attempt {attempt + 1}")
            return json.dumps({"results": results}, ensure_ascii=False)
            
        except Exception as e:
            if attempt < max_retries - 1:
                # Check if it's a 429 error in the exception message
                if "429" in str(e) or "Too Many Requests" in str(e):
                    wait_time = 1 + attempt
                    print(f"⚠️ Brave search 429 error in exception, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ Brave search error on attempt {attempt + 1}: {e}")
                    # For non-429 errors, break and fallback immediately
                    break
            else:
                print(f"❌ Brave search failed after {max_retries} attempts: {e}")
    
    # Fallback to serper search if all brave attempts failed
    print("🔄 Falling back to Serper search...")
    return serper_search(query, num)


def serper_search(query: str, num: int = 5) -> str:
    """
    Web search via Serper API. Requires SERPER_API_KEY.
    Returns JSON with 'results': [{title, link, snippet, source}]
    """
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return json.dumps({"error": "SERPER_API_KEY not set", "results": []})
    try:
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": num},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for it in data.get("organic", [])[:num]:
            results.append(
                {
                    "title": it.get("title"),
                    "link": it.get("link"),
                    "snippet": it.get("snippet"),
                    "source": "organic",
                }
            )
        for it in data.get("news", [])[: max(0, num - len(results))]:
            results.append(
                {
                    "title": it.get("title"),
                    "link": it.get("link"),
                    "snippet": it.get("snippet"),
                    "source": "news",
                }
            )
        return json.dumps({"results": results}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"serper_error: {e}", "results": []})


def _llm_diversify_queries(base_query: str, seen_titles: List[str], k: int, subgraph: Optional[Dict] = None) -> List[str]:
    llm=OpenAI(api_key=openai_api_key, base_url=base_url)
    # llm = LLM(model="gpt-4o-mini", temperature=0.4)
    
    # Prepare subgraph context
    subgraph_context = ""
    if subgraph and isinstance(subgraph, dict):
        nodes = subgraph.get("nodes", [])
        edges = subgraph.get("edges", [])
        if nodes or edges:
            node_names = [node.get("name", "") for node in nodes if isinstance(node, dict)]
            
            # Process edges to understand relationships
            relationships = []
            for edge in edges:
                relationships.append(str(edge))
            
            subgraph_context = f"""
KNOWLEDGE GRAPH CONTEXT:
- Key entities to focus on: {node_names}
- Important relationships: {relationships}
- Generate queries that explore these specific connections and relationships
- Consider how these entities interact and influence each other
- Prioritize queries that reveal causal relationships between the entities
"""
    
    prompt = f"""
You are a finance research strategist. Generate exactly {k} diversified search queries that ALL focus on the same core topic as the user's original question, but explore different angles and aspects.

ORIGINAL USER QUERY: "{base_query}"
{subgraph_context}
Seen titles (avoid duplicating): {seen_titles[:15]}

CORE REQUIREMENT: All generated queries MUST be directly related to the main subject/topic of the user's query. Do NOT deviate to unrelated financial topics.

If user asks about Bitcoin → all queries should be about Bitcoin (price, regulation, adoption, technical analysis, etc.)
If user asks about Fed policy → all queries should be about Federal Reserve policies (interest rates, inflation, market impact, etc.)
If user asks about tech stocks → all queries should be about technology sector stocks (valuations, earnings, growth, etc.)

Diversification Strategy:
1. **Causal factors**: What drives/influences the main topic?
2. **Recent events**: Latest news and developments affecting the topic
3. **Market metrics**: Price, volume, volatility data related to the topic
4. **Regulatory aspect**: Government/regulatory impact on the topic
5. **Historical perspective**: Past trends and patterns for the topic
6. **Expert analysis**: Professional opinions and forecasts on the topic
7. **Comparative analysis**: How the topic compares to alternatives
8. **Future outlook**: Predictions and projections for the topic
9. **Relationship exploration**: If knowledge graph provided, focus on entity relationships and interactions
10. **Cross-impact analysis**: How changes in one entity affect connected entities

CRITICAL FORMAT REQUIREMENTS:
- Return ONLY a valid JSON array of strings
- No additional text, explanations, or markdown
- Exactly {k} strings in the array
- Each query should be 60-120 characters
- Each query must clearly relate to the original topic

Example for "Bitcoin price analysis":
["Bitcoin regulatory news 2024", "Bitcoin institutional adoption trends", "Bitcoin technical analysis support resistance", "Bitcoin vs Ethereum price correlation"]

Example with knowledge graph - Query: "Tesla stock performance", Entities: [Tesla, Federal Reserve, Interest Rates], Relationships: [Federal Reserve controls Interest Rates, Interest Rates affects_valuation Tesla]:
["Tesla stock Federal Reserve interest rate impact", "Tesla valuation sensitivity to monetary policy changes", "Tesla earnings vs interest rate environment correlation", "Tesla stock performance during Fed rate cycles"]

IMPORTANT: If knowledge graph relationships are provided, ensure at least 50% of queries explore these specific entity connections and their market implications.

Return exactly {k} diversified queries as JSON array, ALL focused on: {base_query}"""

    try:
        resp = llm.chat.completions.create(
        model=openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=5000
        )

        # resp_str = str(resp).strip()
        resp_str=resp.choices[0].message.content.strip()
        # Clean up potential markdown or extra text
        if resp_str.startswith('```'):
            lines = resp_str.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('[') and line.endswith(']'):
                    resp_str = line
                    break
        
        ideas = parse_json_response(resp_str, fallback=[])
        if isinstance(ideas, list) and len(ideas) > 0:
            # Ensure we have exactly k queries
            valid_queries = [str(x).strip() for x in ideas if str(x).strip()][:k]
            print(f"📊 Generated {len(valid_queries)} diversified queries from LLM")
            return valid_queries
        else:
            raise ValueError(f"LLM returned invalid format: {type(ideas)}")
    except Exception as e:
        print(f"❌ _llm_diversify_queries failed: {e}, using smart topic-focused fallback")
        # Smart fallback: create topic-focused variations based on query content
        base_lower = base_query.lower()
        fallback_queries = []
        
        # Extract key terms and create focused variations
        if any(term in base_lower for term in ["bitcoin", "btc", "crypto"]):
            main_term = "Bitcoin" if "bitcoin" in base_lower or "btc" in base_lower else "cryptocurrency"
            fallback_queries = [
                f"{main_term} price analysis trends",
                f"{main_term} regulatory developments",
                f"{main_term} institutional adoption",
                f"{main_term} technical analysis"
            ]
        elif any(term in base_lower for term in ["fed", "federal reserve", "interest rate", "monetary policy"]):
            fallback_queries = [
                "Federal Reserve interest rate decisions",
                "Fed policy market impact", 
                "monetary policy economic effects",
                "interest rate forecast analysis"
            ]
        elif any(term in base_lower for term in ["tech", "technology", "nasdaq", "software"]):
            fallback_queries = [
                "technology stocks earnings trends",
                "tech sector valuation analysis",
                "NASDAQ performance indicators",
                "technology companies outlook"
            ]
        elif any(term in base_lower for term in ["government", "policy", "regulation"]):
            fallback_queries = [
                "government policy market impact",
                "regulatory changes industry effects",
                "policy analysis economic implications",
                "government regulation business impact"
            ]
        else:
            # Generic fallback based on the original query
            words = base_query.split()
            if len(words) >= 2:
                key_term = " ".join(words[:2])
                fallback_queries = [
                    f"{key_term} market analysis",
                    f"{key_term} price trends",
                    f"{key_term} industry outlook",
                    f"{key_term} investment potential"
                ]
            else:
                fallback_queries = [
                    f"{base_query} analysis",
                    f"{base_query} trends",
                    f"{base_query} outlook",
                    f"{base_query} forecast"
                ]
        
        return fallback_queries


def multi_search(query: str, context:str,n_wide: int = 3, subgraph: Optional[Dict] = None) -> str:
    """
    Two-step breadth search with parallel execution:
    1) Decompose the query into n_wide diversified finance-aware search queries via LLM.
    2) Run those queries in parallel with brave_search, aggregate and deduplicate results.
    Returns JSON with: {query, sub_queries, n_wide, results:[{... , source_query_idx}]}.
    """
    # robust parse of n_wide
    n = n_wide
    
    print(f"🔍 multi_search: n_wide={n}, generating {n-1} additional queries")
    if subgraph:
        print(f"🧠 Using knowledge graph context with {len(subgraph.get('nodes', []))} nodes")
    
    # Step 1: generate diversified queries (include original as first)
    seen_titles: List[str] = []
    sub_queries: List[str] = [query]
    if n > 1:
        divers = _llm_diversify_queries(query, seen_titles, k=n - 1, subgraph=subgraph)
        
        # Ensure uniqueness and non-empty
        for q in divers:
            qn = (q or "").strip()
            if qn and qn.lower() not in {s.lower() for s in sub_queries}:
                sub_queries.append(qn)
            else:
                print(f"⚠️ Skipped duplicate/empty query: {qn}")
    
    sub_queries = sub_queries[:n]
    print(f"✅ Final sub_queries ({len(sub_queries)}): {sub_queries}")

    # Step 2: parallel search
    agg: List[Dict] = []
    errors: List[Dict] = []
    seen_links = set()

    def _search_one(idx_q: int, qtext: str):
        try:
            raw = brave_search(query=qtext, num=5)
        except Exception as e:
            return idx_q, [], f"tool_call_error:{e}"
        try:
            data = safe_json_loads(raw, fallback={}) if isinstance(raw, str) else raw
            res = data.get("results") or []
            err = data.get("error")
            return idx_q, res, err
        except Exception as e:
            return idx_q, [], f"json_parse_error:{e}"

    with ThreadPoolExecutor(max_workers=min(len(sub_queries), 5)) as executor:
        futures = [executor.submit(_search_one, i, q) for i, q in enumerate(sub_queries)]
        for future in as_completed(futures):
            idx_q, res, err = future.result()
            if err:
                errors.append({"query_idx": idx_q, "error": err})
            for item in res:
                link = item.get("link")
                if link and link not in seen_links:
                    seen_links.add(link)
                    item["source_query_idx"] = idx_q
                    agg.append(item)

    print(f"🎯 multi_search: aggregated {len(agg)} unique results from {len(sub_queries)} queries")
    
    return json.dumps({
        "query": query,
        "sub_queries": sub_queries,
        "n_wide": n,
        "results": agg,
        "errors": errors
    }, ensure_ascii=False)
